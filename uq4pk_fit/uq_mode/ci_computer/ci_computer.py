
import numpy as np
import ray

from ..linear_model import CredibleRegion, LinearModel
from ..evaluation import AffineEvaluationFunctional, AffineEvaluationMap
from ..optimization import ECOS, SCS, SLSQP, SOCP, socp_solve, socp_solve_remote
from .credible_intervals import CredibleInterval
from .progress_bar import ProgressBar


RTOL = 0.001     # relative tolerance for the cost-constraint
RACC = 1e-3      # relative accuracy for optimization solvers.
DEFAULT_SOLVER = "ecos"


class CIComputer:
    """
    Manages the computation of credible intervals from evaluation maps.

    Given a linear model and an evaluation map, the CI computer solves for each evaluation object (loss, x, phi)
    the optimization problems
    min_z / max_z w^\top z s.t. AU z = b - A v, ||C z - d||_2^2 <= e.
    and returns the credible interval [phi(z_min), phi(z_max)].
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, aemap: AffineEvaluationMap, scale: float,
                 options: dict):
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._aemap = aemap
        cost_map = model.cost(x_map)
        self._ctol = RTOL * cost_map

        self._cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)
        # Read options.
        if options is None:
            options = {}
        self._use_ray = options.setdefault("use_ray", True)
        self._num_cpus = options.setdefault("num_cpus", 7)
        solver_name = options.setdefault("solver", DEFAULT_SOLVER)
        if solver_name == "slsqp":
            self._optimizer = SLSQP(ftol=1e-8)
        elif solver_name == "ecos":
            abstol = RACC * scale
            self._optimizer = ECOS(abstol=abstol)
        elif solver_name == "scs":
            self._optimizer = SCS()
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

    def compute_all(self) -> CredibleInterval:
        """
        Computes all credible intervals.

        :returns: Object of type :py:class:`CredibleInterval`
        """
        # Initialize lists for storing the FCI-values.
        out_lower_list = []
        out_upper_list = []
        # For every coordinate, compute the value of the lower and upper bounds of the kernel localization functional
        # over the credible region.
        cicounter = 0       # counts number of computed credible intervals
        for aefun in self._aemap.aef_list:
            cicounter += 1
            # Compute the i-th filtered credible interval
            out_lower, out_upper = self._compute_interval(aefun)
            self._print_status(cicounter=cicounter)
            out_lower_list.append(out_lower)
            out_upper_list.append(out_upper)
        if self._use_ray:
            print("\n" + "Starting computation...")
            self._pb.print_until_done()
            out_lower_list_result = ray.get(out_lower_list)
            out_upper_list_result = ray.get(out_upper_list)
        else:
            out_lower_list_result = out_lower_list
            out_upper_list_result = out_upper_list
        ray.shutdown()
        # The Ray results are now converted to an array. The j-th row of the array corresponds to the credible interval
        # associated to the j-th window-frame pair.
        phi_lower = np.array([out[0] for out in out_lower_list_result])
        phi_upper = np.array([out[0] for out in out_upper_list_result])
        times_lower = np.array([out[1] for out in out_lower_list_result])
        times_upper = np.array([out[1] for out in out_upper_list_result])
        times = times_lower + times_upper
        time_avg = np.mean(np.array(times))
        assert np.all(phi_lower <= phi_upper + 1e-8)
        credible_interval = CredibleInterval(phi_lower=phi_lower, phi_upper=phi_upper, time_avg=time_avg)

        return credible_interval

    # PROTECTED

    def _compute_interval(self, aefun: AffineEvaluationFunctional):
        """
        Computes the minimum and maximum inside the credible region of the quantity
        of interest defined by an affine evaluation functional.

        :param aefun: The affine evaluation functional.

        :returns: qoi_low, x_low, qoi_up, x_up
            - qoi_low: The minimum of the quantity of interest, aefun.phi.
            - x_low: The corresponding minimizer.
            - qoi_up: The maximum of the quantity of interest.
            - x_up: The corresponding maximizer.
        """
        # Create SOCP problem
        socp = self._create_socp(aefun=aefun)
        # Compute minimizer/maximizer
        z0 = aefun.z0
        if self._use_ray:
            out_low = socp_solve_remote.remote(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol,
                                               qoi=aefun.phi, actor=self._actor, mode="min")
            out_up = socp_solve_remote.remote(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol,
                                              qoi=aefun.phi, actor=self._actor, mode="max")
        else:
            out_low = socp_solve(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol, qoi=aefun.phi,
                                 mode="min")
            out_up = socp_solve(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol, qoi=aefun.phi,
                                mode="max")
        return out_low, out_up

    def _create_socp(self, aefun: AffineEvaluationFunctional) -> SOCP:
        """
        Creates the SOCP for the computation of the generalized credible interval.
        The constraints
        ||T x - d_tilde||_2^2 <= e_tilde,
        A x = b,
        x >= lb

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
        c_z = self._cregion.t @ u
        g = self._cregion.d_tilde - self._cregion.t @ v
        p_z, t_z0 = np.linalg.qr(c_z, mode="complete")
        k = aefun.zdim
        t_z = t_z0[:k, :]
        p_z1 = p_z[:, :k]
        p_z2 = p_z[:, k:]
        d_tilde_z = p_z1.T @ g
        e_tilde_z = self._cregion.e_tilde - np.sum(np.square(p_z2.T @ g))

        # Reformulate the equality constraint for z.
        if self._cregion.a is not None:
            a_new = self._cregion.a @ u
            b_new = self._cregion.b - self._cregion.a @ v
            # If equality constraint does not satisfy constraint qualification, it is removed.
            satisfies_cq = (np.linalg.matrix_rank(a_new) >= a_new.shape[0])
            if not satisfies_cq:
                a_new = None
                b_new = None
        else:
            a_new = None
            b_new = None

        # Reformulate the lower-bound constraint for z.
        lb_z = aefun.lb_z(self._cregion.lb)

        # Create SOCP instance
        socp = SOCP(w=w, a=a_new, b=b_new, c=t_z, d=d_tilde_z, e=e_tilde_z, lb=lb_z)
        return socp

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
