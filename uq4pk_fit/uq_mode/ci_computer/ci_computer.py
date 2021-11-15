
from math import sqrt, log
import numpy as np
import ray
import time
from typing import Literal

from ..linear_model import LinearModel
from ..evaluation import AffineEvaluationFunctional, AffineEvaluationMap
from ..optimization import ECOS, SLSQP, SOCP, socp_solve, socp_solve_remote


RTOL = 0.01     # relative tolerance for the cost-constraint
FTOL = 1e-5


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
        solver_name = options.setdefault("solver", "slsqp")
        if solver_name == "slsqp":
            self._optimizer = SLSQP(ftol=FTOL)
            print("Using SLSQP solver.")
        elif solver_name == "ecos":
            self._optimizer = ECOS(scale=self._cost_map)
            print("Using ECOS solver.")
        else:
            raise KeyError(f"Unknown solver '{solver_name}'.")

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
        print(" ")
        if self._use_ray:
            ray.init(num_cpus=self._num_cpus)
        t_list = []
        t_avg = "undefined"
        counter = 0
        for aefun in self._aemap.aef_list:
            counter += 1
            print("\r", end="")
            print(f"Computing credible interval {counter}/{self._aemap.size} (avg {t_avg} s)",
                  end=" ")
            # Compute the lower bound for the local credible interval with respect to the i-th localization functional.
            t0 = time.time()
            x_lower = self._minimize(aefun)
            x_upper = self._maximize(aefun)
            x_lower_list.append(x_lower)
            x_upper_list.append(x_upper)
            t = time.time() - t0
            t_list.append(t)
            t_avg = np.mean(np.array(t_list))
        print("Collecting ...", end="")
        if self._use_ray:
            x_lower_list_result = ray.get(x_lower_list)
            x_upper_list_result = ray.get(x_upper_list)
        else:
            x_lower_list_result = x_lower_list
            x_upper_list_result = x_upper_list
        ray.shutdown()
        print(" done.")
        # The Ray results are now converted to an array. The j-th row of the array corresponds to the credible interval
        # associated to the j-th window-frame pair.
        x_lower = np.concatenate(x_lower_list_result)
        x_upper = np.concatenate(x_upper_list_result)
        assert np.all(x_lower <= x_upper + 1e-8)
        return np.column_stack([x_lower, x_upper])

    def _minimize(self, aefun: AffineEvaluationFunctional):
        """
        Computes the minimal value of the quantity of interest, with respect to the loss function and the constraints.
        """
        minimum = self._compute(aefun, 0)
        return minimum

    def _maximize(self, aefun: AffineEvaluationFunctional):
        """
        Computes the maximal value of the quantity of interest, with respect to the loss function and the constraints.
        """
        maximum = self._compute(aefun, 1)
        return maximum

    # PROTECTED

    def _compute(self, aefun: AffineEvaluationFunctional, minmax: Literal[0, 1]):
        """
        Depending on the value of minmax, computes the minimum or maximum inside the credible region of the quantity
        of interest defined by an affine evaluation functional.

        :param aefun:
        :param minmax: If 0, then the minimum is returned. If 1, then the maximum is returned.
        """
        # Create SOCP problem
        socp = self._create_socp(aefun=aefun, minmax=minmax)
        # Compute minimizer/maximizer
        z0 = aefun.z0
        if self._use_ray:
            qoi = socp_solve_remote.remote(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol,
                                           qoi=aefun.phi)
        else:
            z_opt = socp_solve(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol)
            qoi = aefun.phi(z_opt)
        return qoi

    def _create_socp(self, aefun: AffineEvaluationFunctional, minmax: int) -> SOCP:
        """
        Creates the SOCP for the computation of the generalized credible interval.
        The constraints
        ||Hx - y||_2^2 + ||R(x - m)||_2^2 <= e
        A x = b
        x >= lb

        are reformulated in terms of z, where x = U z + v:
        ||C z - d||_2^2 <= e,
        A_new z = b_new,
        z >= lb_z,
        where
            A_new = A M
            b_new = b - A v
            C = [C1; C2], d = [d1; d2]
            C1 = Q H U
            C2 = R U
            d1 = Q (y - H v)
            d2 = R (m - v)
            e = 2 * (cost_map + k_alpha)
            lb_z = [depends on affine evaluation functional]
        """
        w = aefun.w
        u = aefun.u
        v = aefun.v
        a = self._model.a
        b = self._model.b
        x0 = aefun.x(aefun.z0)
        h = self._model.h
        y = self._model.y
        q = self._model.q
        r = self._model.r
        m = self._model.m
        lb = self._model.lb
        cost_map = self._cost_map
        k_alpha = self._k_alpha
        # check that x_map satisfies credibility constraint
        credibility = cost_map + k_alpha - 0.5 * np.sum(np.square(q.fwd(h @ x0 - y))) \
                      - 0.5 * np.sum(np.square(r.fwd(x0 - m)))
        assert credibility >= - self._ctol
        if a is not None:
            a_new = a @ u
            b_new = b - a @ v
            # If equality constraint does not satisfy constraint qualification, it is removed.
            satisfies_cq = (np.linalg.matrix_rank(a_new) >= a_new.shape[0])
            if not satisfies_cq:
                a_new = None
                b_new = None
        else:
            a_new = None
            b_new = None
        c1 = q.fwd(h @ u)
        c2 = r.fwd(u)
        c = np.concatenate([c1, c2], axis=0)
        d1 = q.fwd(y - h @ v)
        d2 = r.fwd(m - v)
        d = np.concatenate([d1, d2], axis=0)
        e = 2 * (cost_map + k_alpha)
        lb_z = aefun.lb_z(lb)
        # Create SOCP instance
        socp = SOCP(w=w, a=a_new, b=b_new, c=c, d=d, e=e, lb=lb_z, minmax=minmax)
        self._check_socp(aefun, socp)
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

    def _check_socp(self, aefun: AffineEvaluationFunctional, socp: SOCP):
        """
        Rough check that the SOCP was initialized correctly.
        """
        m = 5
        n_z = socp.w.size
        for i in range(m):
            z_test = np.random.randn(n_z)
            x_test = aefun.x(z_test)
            cost_x = self._model.cost(x_test)
            cost_z = 0.5 * np.sum(np.square(socp.c @ z_test - socp.d))
            assert np.isclose(cost_x, cost_z)
