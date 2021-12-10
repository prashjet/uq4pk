
from math import sqrt, log
import numpy as np
import ray

from ..linear_model import LinearModel
from ..evaluation import AffineEvaluationFunctional, AffineEvaluationMap
from ..optimization import ECOS, SLSQP, IPOPT, SOCP, socp_solve, socp_solve_remote
from .progress_bar import ProgressBar


RTOL = 0.01     # relative tolerance for the cost-constraint
FTOL = 1e-5
DEFAULT_SOLVER = "ecos"


class CIComputer:
    """
    Manages the computation of credible intervals from evaluation maps.

    Given a linear model and an evaluation map, the CI computer solves for each evaluation object (loss, x, phi)
    the optimization problems
    min_z / max_z w^\top z s.t. AU z = b - A v, ||C z - d||_2^2 <= e.
    and returns the credible interval [phi(z_min), phi(z_max)].
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, aemap: AffineEvaluationMap, options: dict):
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._aemap = aemap
        self._model = model
        self._costf = model.cost
        self._costg = model.cost_grad
        self._cost_map = model.cost(x_map)
        self._tau = sqrt(16 * log(3 / alpha) / self._dim)
        self._k_alpha = self._dim * (self._tau + 1)
        self._ctol = RTOL * self._k_alpha
        # Read options.
        if options is None:
            options = {}
        self._use_ray = options.setdefault("use_ray", True)
        self._num_cpus = options.setdefault("num_cpus", 8)
        solver_name = options.setdefault("solver", DEFAULT_SOLVER)
        if solver_name == "slsqp":
            self._optimizer = SLSQP(ftol=FTOL)
        elif solver_name == "ipopt":
            self._optimizer = IPOPT(ftol=FTOL)
        elif solver_name == "ecos":
            self._optimizer = ECOS(scale=self._cost_map)
        else:
            raise KeyError(f"Unknown solver '{solver_name}'.")
        if solver_name != DEFAULT_SOLVER:
            print(f"Using {solver_name} solver.")
        self._cino = len(self._aemap.aef_list)  # number of credible intervals
        self._optno = 2 * self._cino            # number of optimization problems to solve
        # Setting up progress bar
        if self._use_ray:
            if self._use_ray:
                ray.shutdown()
                ray.init(num_cpus=self._num_cpus)
            self._pb = ProgressBar(self._optno)
            self._actor = self._pb.actor

        # Preprocessing: Assemble the matrices for the SOCP problem and compute QR decomposition of C.
        self._t, self._d_tilde, self._e_tilde, self._a, self._b, self._lb = self._preprocessing()

    def compute_all(self) -> np.ndarray:
        """
        Computes all credible intervals.
        :return: Of shape (n, 2)
            The j-th row of the array corresponds to the lower and upper bound for the credible interval that is
            associated to the j-th coordinate.
        """
        x_lower_list = []
        x_upper_list = []
        # For every coordinate, compute the value of the lower and upper bounds of the kernel localization functional
        # over the credible region.
        cicounter = 0       # counts number of computed credible intervals
        for aefun in self._aemap.aef_list:
            cicounter += 1
            # Compute the i-th filtered credible interval
            x_lower, x_upper = self._compute_interval(aefun)
            self._print_status(cicounter=cicounter)
            x_lower_list.append(x_lower)
            x_upper_list.append(x_upper)
        if self._use_ray:
            print("\n" + "Starting computation...")
            self._pb.print_until_done()
            x_lower_list_result = ray.get(x_lower_list)
            x_upper_list_result = ray.get(x_upper_list)
        else:
            x_lower_list_result = x_lower_list
            x_upper_list_result = x_upper_list
        ray.shutdown()
        # The Ray results are now converted to an array. The j-th row of the array corresponds to the credible interval
        # associated to the j-th window-frame pair.
        x_lower = np.concatenate(x_lower_list_result)
        x_upper = np.concatenate(x_upper_list_result)
        assert np.all(x_lower <= x_upper + 1e-8)
        return np.column_stack([x_lower, x_upper])

    # PROTECTED

    def _preprocessing(self):
        """
        Computes all entities necessary for the formulation of the constraints
        ||C f - d||_2^2 <= e,
        A f = b
        f >= lb.
        We use a QR decomposition to reduce the cone constraint to
        ||T f - d_tilde||_2^2 <= e_tilde,
        where R is invertible upper triangular.

        :return: t, d_tilde, e_tilde, a, b, lb.
            - t: An invertible upper triangular matrix of shape (n, n).
            - d_tilde: An n-vector.
            - e_tilde: A nonnegative float.
            - a: A matrix of shape (c, n).
            - b: A c-vector.
            - lb: An n-vector, representing the lower bound.
        """
        a = self._model.a
        b = self._model.b
        h = self._model.h
        y = self._model.y
        q = self._model.q
        r = self._model.r
        m = self._model.m
        lb = self._model.lb
        n = self._model.n

        # Assemble the matrix C, the vector d and the RHS e.
        cost_map = self._cost_map
        k_alpha = self._k_alpha
        c1 = q.fwd(h)
        c2 = r.mat
        c = np.concatenate([c1, c2], axis=0)
        d1 = q.fwd(y)
        d2 = r.fwd(m)
        d = np.concatenate([d1, d2], axis=0)
        e = 2 * (cost_map + k_alpha)

        # Compute the QR decomposition of C.
        p, t0 = np.linalg.qr(c, mode="complete")
        t = t0[:n, :]
        p1 = p[:, :n]
        p2 = p[:, n:]

        # Compute d_tilde and e_tilde.
        d_tilde = p1.T @ d
        e_tilde = e - np.sum(np.square(p2.T @ d))

        # Return everything in the right-order.
        return t, d_tilde, e_tilde, a, b, lb

    def _compute_interval(self, aefun: AffineEvaluationFunctional):
        """
        Computes the minimum and maximum inside the credible region of the quantity
        of interest defined by an affine evaluation functional.

        :param aefun:
        """
        # Create SOCP problem
        socp = self._create_socp(aefun=aefun)
        # Compute minimizer/maximizer
        z0 = aefun.z0
        if self._use_ray:
            qoi_low = socp_solve_remote.remote(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol,
                                               qoi=aefun.phi, actor=self._actor, mode="min")
            qoi_up = socp_solve_remote.remote(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol,
                                              qoi=aefun.phi, actor=self._actor, mode="max")
        else:
            qoi_low = socp_solve(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol, qoi=aefun.phi,
                                 mode="min")
            qoi_up = socp_solve(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol, qoi=aefun.phi,
                                mode="max")
        return qoi_low, qoi_up

    def _create_socp(self, aefun: AffineEvaluationFunctional) -> SOCP:
        """
        Creates the SOCP for the computation of the generalized credible interval.
        The constraints
        ||T f - d_tilde||_2^2 <= e_tilde,
        A f = b,
        f >= lb

        are reformulated in terms of z, where x = U z + v:
        ||T_z z - d_z||_2^2 <= e_z,
        A_z z = b_z,
        z >= lb_z,
        where
            d_z = P_z1.T d_tilde.
            e = e_z - ||P_z1.T d_tilde||_2^2
            P_z [T_z; 0] is the QR decomposition of T U.
            A_z = A U
            b_z= b - A v
            lb_z = [depends on affine evaluation functional]
        """
        w = aefun.w
        u = aefun.u
        v = aefun.v

        # Reformulate the cone constraint for z.
        c_z = self._t @ u
        g = self._d_tilde - self._t @ v
        p_z, t_z0 = np.linalg.qr(c_z, mode="complete")
        k = aefun.zdim
        t_z = t_z0[:k, :]
        p_z1 = p_z[:, :k]
        p_z2 = p_z[:, k:]
        d_tilde_z = p_z1.T @ g
        e_tilde_z = self._e_tilde - np.sum(np.square(p_z2.T @ g))

        # Reformulate the equality constraint for z.
        if self._a is not None:
            a_new = self._a @ u
            b_new = self._b - self._a @ v
            # If equality constraint does not satisfy constraint qualification, it is removed.
            satisfies_cq = (np.linalg.matrix_rank(a_new) >= a_new.shape[0])
            if not satisfies_cq:
                a_new = None
                b_new = None
        else:
            a_new = None
            b_new = None

        # Reformulate the lower-bound constraint for z.
        lb_z = aefun.lb_z(self._lb)

        # Create SOCP instance
        socp = SOCP(w=w, a=a_new, b=b_new, c=t_z, d=d_tilde_z, e=e_tilde_z, lb=lb_z)
        return socp

    def _cost_constraint(self, x: np.ndarray) -> float:
        """
        The cost constraint is
        c(x) >= 0, where c(x) = J(x_map) + k_\alpha - J(x),
        where J is the MAP cost functional, and k_\alpha = N * (\tau_\alpha + 1).
        """
        c = self._cost_map + self._k_alpha - self._costf(x)
        return c

    def _cost_constraint_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the cost constraint function. That is

        :math:`\\nabla c(x) = - \\nabla J(x).
        """
        return - self._costg(x)

    @staticmethod
    def _negative(v: np.ndarray):
        """
        Returns the negative part of a vector v.

        For example, the vector :math:`v = [-1, 0, 3]` has negative part :math:`v^- = [1, 0, 0]`.
        """
        vminus = -v
        vneg = vminus.clip(min=0.)
        return vneg

    def _print_status(self, cicounter: int):
        """
        Displays the current status of the computation.

        :param cicounter: Number of current credible interval.
        """
        if self._use_ray:
            print("\r", end="")
            print(f"Preparing credible interval {cicounter}/{self._cino}.",
                  end=" ")
        else:
            print("\r", end="")
            print(f"Computing credible interval {cicounter}/{self._cino}.",
                  end=" ")
