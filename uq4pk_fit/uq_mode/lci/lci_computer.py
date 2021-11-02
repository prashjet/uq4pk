"""
Contains class "LCIComputer".
"""

from math import log, sqrt
import numpy as np
import time
from typing import List

from ..optimization import NullConstraint, OptimizationProblem, SLSQP
from ..partition import Partition
from .local_value import LocalValue


class LCIComputer:

    def __init__(self, alpha: float, cost: callable, costgrad: callable, x_map: np.ndarray, partition: Partition, 
                 lb: np.ndarray = None):
        self._partition = partition
        self._npartition = partition.size
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._costf = cost
        self._costg = costgrad
        self._cost_map = cost(x_map)
        self._tau = sqrt(16 * log(3 / alpha) / self._dim)
        self._k_alpha = self._dim * (self._tau + 1)
        self.RTOL = 0.01  # 1% relative tolerance for optimization
        self.ctol = self.RTOL * self._k_alpha
        self._lb = lb
        self._optimizer = SLSQP()
        # make the partition matrix that maps a vector fo size self._npartition to a vector of size self._dim.
        self._pmat = np.zeros((self._dim, self._npartition))
        for i in range(self._npartition):
            omega_i = partition.element(i)
            self._pmat[omega_i, i] = 1
        assert np.linalg.matrix_rank(self._pmat) == self._npartition

    def compute(self):
        """
        Computes the local credible intervals.
        :return: (n, 2) array
            The j-th row of the array corresponds to the lower and upper bound for the credible interval that is
            associated to the j-th coordinate by LMCIComputer.window_function.
        """
        x_lower_list = []
        x_upper_list = []
        # For every coordinate, compute the value of the lower and upper bounds of the kernel localization functional
        # over the credible region.
        print(" ")
        t0 = time.time()
        for i in range(self._npartition):
            t = time.time() - t0
            t0 = time.time()
            print("\r", end="")
            print(f"Computing local credible interval {i + 1}/{self._npartition} ({t} s)", end=" ")
            # Compute local credible interval with respect to i-th partition element
            omega_i = self._partition.element(i)
            lvalue = LocalValue(x_map=self._x_map, indices=omega_i)
            x_lower = self._minimize(lvalue)
            x_upper = self._maximize(lvalue)
            x_lower_list.append(x_lower)
            x_upper_list.append(x_upper)
        # The results are now converted to an array. The j-th row of the array corresponds to the credible interval
        # associated to the j-th window-frame pair.
        x_lower = self._translate(x_lower_list)
        x_upper = self._translate(x_upper_list)
        lci_arr = np.column_stack((x_lower, x_upper))
        # The enlarged array is then returned to the calling function (and then, to the user).
        return lci_arr

    def _minimize(self, lvalue: LocalValue):
        """
        Computes the minimal value of the quantity of interest, with respect to the loss function and the constraints.
        :return: float
        """
        minimum = self._compute(lvalue, 0)
        return minimum

    def _maximize(self, lvalue: LocalValue):
        """
        Computes the maximal value of the quantity of interest, with respect to the loss function and the constraints.
        :return: float
        """
        maximum = self._compute(lvalue, 1)
        return maximum

    # PROTECTED

    def _compute(self, lvalue: LocalValue, minmax: int, plot=False) -> float:
        """
        Computes the quantity of interest.
        """
        # Create OptimizationProblem object
        opt_problem = self._create_optimization_problem(lvalue, minmax)
        # Compute minimizer/maximizer
        xi = self._optimizer.optimize(opt_problem)
        # Check that z is truly an optimizer
        self._check_optimality(xi, opt_problem)
        return lvalue.x(xi)[lvalue._ind]

    def _create_optimization_problem(self, lvalue: LocalValue, minmax: int):
        """
        Creates optimization problem
        """
        # Setup objective
        if minmax == 0:
            loss_fun = lvalue.loss_fun
            loss_grad = lvalue.loss_grad
        else:
            def loss_fun(xi):
                return - lvalue.loss_fun(xi)

            def loss_grad(xi):
                return - lvalue.loss_grad(xi)
        # Transform credibility constraint
        cred_constraint = lvalue.transform_nonlinear_constraint(fun=self._crediblity_constraint_fun,
                                                                jac=self._credibility_constraint_jac, type="ineq")
        # Transform lower bound constraint
        lb = lvalue.lower_bound(self._lb)
        z0 = lvalue.initial_value
        opt_problem = OptimizationProblem(loss_fun=loss_fun, loss_grad=loss_grad, start=z0, eqcon=NullConstraint(),
                                          incon=cred_constraint, lb=lb)
        return opt_problem

    def _crediblity_constraint_fun(self, x):
        """
        The constraint over z.
        phi(x) <= phi(x_map) + lvl
        <=> phi(x_map) + lvl - phi(x) >= 0
        """
        c = self._cost_map + self._k_alpha - self._costf(x)
        return c

    def _credibility_constraint_jac(self, x):
        jac = - self._costg(x)
        return jac

    def _translate(self, xi_list: List[np.ndarray]) -> np.ndarray:
        """
        Enlarges a vector of size :py:attr:`self._npartition` to a vector of size :py:attr:`self._dim`.

        """
        x = np.zeros((self._dim, ))
        for i in range(self._npartition):
            omega_i = self._partition.element(i)
            x[omega_i] = xi_list[i]
        return x

    @staticmethod
    def _negative(v):
        vminus = -v
        vneg = vminus.clip(min=0.)
        return vneg

    def _check_optimality(self, z: np.ndarray, problem: OptimizationProblem):
        """
        Checks whether z is truly an optimizer.
        :param z: (k,) array_like
        """
        if problem.lb is None:
            # If there are no lower bound constraints, the optimizer should be at the boundary of the credible region.
            # This means that, in this case, the inequality constraint should hold with equality.
            region_error = np.abs(problem.incon.fun(z))
        else:
            # If there are lower bound constraints, then the optimizer either lies at the boundary of the credible region,
            # or one of the lower bound constraints is active.
            if self._lower_bound_active(z, problem):
                # In any case, the optimizer must lie inside the credible region.
                region_error = self._negative(problem.incon.fun(z))
            else:
                region_error = np.abs(problem.incon.fun(z))
        if not isinstance(problem.eqcon, NullConstraint):
            # If an equality constraint is active, we also have to account for that.
            eqcon_error = np.linalg.norm(problem.eqcon.fun(z))
        else:
            eqcon_error = 0.
        constraint_error = region_error + eqcon_error
        if constraint_error > self.ctol:
            print("WARNING: Solver was not able to satisfy all constraints.")
            print(f"(tolerance = {self.ctol}, error = {constraint_error}")

    @staticmethod
    def _lower_bound_active(z, problem):
        """
        Checks whether the lower bound is active at z.
        :param z: (k,) array_like
        :return: bool
        """
        distance_to_lower_bound = np.min(z - problem.lb)
        if np.isclose(distance_to_lower_bound, 0):
            is_active = True
        else:
            is_active = False
        return is_active