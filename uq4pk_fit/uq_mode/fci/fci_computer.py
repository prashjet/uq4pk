
from math import sqrt, log
import numpy as np
import time
import ray

from ..filter import FilterFunction
from ..optimization import ECOS, SLSQP, SOCP, socp_solve, socp_solve_remote
from .filtered_value import FilterValue
from ..linear_model import LinearModel


RTOL = 0.01     # relative tolerance for the cost-constraint
FTOL = 1e-5


class FCIComputer:
    """
    Superclass for computation of quantity of interests.
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction, options: dict):
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._ffunction = ffunction
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
        elif solver_name == "ecos":
            self._optimizer = ECOS(scale=self._cost_map)
            print("Using the ECOS solver to solve the SOC problems. WARNING: THIS CAN BE EXTREMELY SLOW.")
        else:
            raise KeyError(f"Unknown solver '{solver_name}'.")

    def compute(self):
        """
        Computes the kernel-local credible intervals.
        :return: (n, 2) array
            The j-th row of the array corresponds to the lower and upper bound for the credible interval that is
            associated to the j-th coordinate by LMCIComputer.window_function.
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
        for i in range(self._ffunction.size):
            print("\r", end="")
            print(f"Computing filtered credible interval {i + 1}/{self._ffunction.size} (avg {t_avg} s)",
                  end=" ")
            # Compute the lower bound for the local credible interval with respect to the i-th localization functional.
            t0 = time.time()
            filter = self._ffunction.filter(i)
            fvalue = FilterValue(x_map=self._x_map, filter=filter)
            x_lower = self._minimize(fvalue)
            x_upper = self._maximize(fvalue)
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
        x_lower = np.array(x_lower_list_result)
        x_upper = np.array(x_upper_list_result)
        assert np.all(x_lower <= x_upper + 1e-8)
        # The vectors are enlarged so that they are of the same dimension as the estimate:
        x_lower_enlarged = self._ffunction.enlarge(x_lower)
        x_upper_enlarged = self._ffunction.enlarge(x_upper)
        lci_arr = np.column_stack((x_lower_enlarged, x_upper_enlarged))
        # The enlarged array is then returned to the calling function (and then, to the user).
        return lci_arr

    def _minimize(self, fvalue: FilterValue):
        """
        Computes the minimal value of the quantity of interest, with respect to the loss function and the constraints.
        :return: float
        """
        minimum = self._compute(fvalue, 0)
        return minimum

    def _maximize(self, fvalue: FilterValue):
        """
        Computes the maximal value of the quantity of interest, with respect to the loss function and the constraints.
        :return: float
        """
        maximum = self._compute(fvalue, 1)
        return maximum

    # PROTECTED

    def _compute(self, fvalue: FilterValue, minmax):
        """
        Computes the quantity of interest.
        """
        # Create SOCP problem
        socp = self._create_socp(fvalue=fvalue, minmax=minmax)
        # Compute minimizer/maximizer
        z0 = fvalue.initial_value
        if self._use_ray:
            phi = socp_solve_remote.remote(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol)
        else:
            phi = socp_solve(socp=socp, start=z0, optimizer=self._optimizer, ctol=self._ctol)
        return phi

    def _create_socp(self, fvalue: FilterValue, minmax: int):
        """
        Creates the SOCP for the computation of the filtered credible interval.

        The SOCP is
        min / max_z w @ z
        s.t. A_new z = b_new, ||C z - d||_2^2 <= e, z >= lb_z,
        where
            A_new = A dx_dz
            b_new = b - A (x_map - dx_dz z_map)
            C = [C1; C2], d = [d1; d2]
            C1 = Q H dx_dz
            C2 = R dx_dz
            d1 = Q (y - H (x_map - dx_dz z_map))
            d2 = R (m - (x_map - dx_dz z_map))
            e = 2 * (cost_map + k_alpha)
            l_z = lb[fvalue.indices]
        """
        w = fvalue.weights
        a = self._model.a
        b = self._model.b
        dx_dz = fvalue.dx_dz
        x_map = self._x_map
        z_map = fvalue.z_map
        # z_map must satisfy x(z_map) = x_map
        assert np.isclose(fvalue.x(z_map), x_map).all()
        h = self._model.h
        y = self._model.y
        q = self._model.q
        r = self._model.r
        m = self._model.m
        lb = self._model.lb
        cost_map = self._cost_map
        k_alpha = self._k_alpha
        # check that x_map satisfies credibility constraint
        credibility = cost_map + k_alpha - 0.5 * np.sum(np.square(q.fwd(h @ x_map - y))) \
                      - 0.5 * np.sum(np.square(r.fwd(x_map - m)))
        assert credibility >= - self._ctol
        if a is not None:
            a_new = a @ dx_dz
            b_new = b - a @ (x_map - dx_dz @ z_map)
            # If equality constraint does not satisfy constraint qualification, it is removed.
            satisfies_cq = (np.linalg.matrix_rank(a_new) >= a_new.shape[0])
            if not satisfies_cq:
                a_new = None
                b_new = None
        else:
            a_new = None
            b_new = None
        c1 = q.fwd(h @ dx_dz)
        c2 = r.fwd(dx_dz)
        c = np.concatenate([c1, c2], axis=0)
        d1 = q.fwd(y - h @ (x_map - dx_dz @ z_map))
        d2 = r.fwd(m - (x_map - dx_dz @ z_map))
        d = np.concatenate([d1, d2], axis=0)
        e = 2 * (cost_map + k_alpha)
        lb_z = fvalue.lower_bound(lb)
        # Create SOCP instance
        socp = SOCP(w=w, a=a_new, b=b_new, c=c, d=d, e=e, lb=lb_z, minmax=minmax)
        self._check_socp(fvalue, socp)
        return socp

    def _cost_constraint(self, x):
        c = self._cost_map + self._k_alpha - self._costf(x)
        return c

    def _cost_constraint_grad(self, x):
        return - self._costg(x)

    @staticmethod
    def _negative(v):
        vminus = -v
        vneg = vminus.clip(min=0.)
        return vneg

    def _check_socp(self, fvalue: FilterValue, socp: SOCP):
        m = 5
        n_z = socp.w.size
        for i in range(m):
            z_test = np.random.randn(n_z)
            x_test = fvalue.x(z_test)
            cost_x = self._model.cost(x_test)
            cost_z = 0.5 * np.sum(np.square(socp.c @ z_test - socp.d))
            assert np.isclose(cost_x, cost_z)
