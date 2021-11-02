
from math import sqrt, log
import numpy as np
import time

from ..filter import FilterFunction
from ..optimization import SLSQP, OptimizationProblem
from .filtered_value import FilterValue
from ..linear_model import LinearModel


class FCIComputer:
    """
    Superclass for computation of quantity of interests.
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction):
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
        self.RTOL = 0.01  # 1% relative tolerance for optimization
        self.ctol = self.RTOL * self._k_alpha
        self._optimizer = SLSQP()

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
        t0 = time.time()
        for i in range(self._ffunction.size):
            t = time.time() - t0
            t0 = time.time()
            print("\r", end="")
            print(f"Computing filtered credible interval {i + 1}/{self._ffunction.size} ({t} s)", end=" ")
            # Compute the lower bound for the local credible interval with respect to the i-th localization functional.
            filter = self._ffunction.filter(i)
            fvalue = FilterValue(x_map=self._x_map, filter=filter)
            x_lower = self._minimize(fvalue)
            x_upper = self._maximize(fvalue)
            x_lower_list.append(x_lower)
            x_upper_list.append(x_upper)
        # The results are now converted to an array. The j-th row of the array corresponds to the credible interval
        # associated to the j-th window-frame pair.
        x_lower = np.array(x_lower_list)
        x_upper = np.array(x_upper_list)
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
        problem = self._create_optimization_problem(fvalue=fvalue, minmax=minmax)
        # Compute minimizer/maximizer
        z0 = fvalue.initial_value
        z = self._optimizer.optimize(problem)
        assert z.size == z0.size
        # check that minimizer satisfies constraints
        #problem.check_constraints(z, tol=self.ctol)
        # return filter value
        phi = fvalue.phi(z)
        return phi

    def _cost_constraint(self, x):
        c = self._cost_map + self._k_alpha - self._costf(x)
        return c

    def _cost_constraint_grad(self, x):
        return - self._costg(x)

    def _create_optimization_problem(self, fvalue: FilterValue, minmax: int) -> OptimizationProblem:
        """
        Translates the linear model for x
        """
        if minmax == 0:
            lossfun = fvalue.phi
            lossgrad = fvalue.phi_grad
        else:
            def lossfun(z):
                return - fvalue.phi(z)
            def lossgrad(z):
                return - fvalue.phi_grad(z)
        cost_constraint = fvalue.transform_nonlinear_constraint(fun=self._cost_constraint, jac=self._cost_constraint_grad,
                                                                type="ineq")
        equality_constraint = fvalue.transform_linear_constraint(a=self._model.a, b=self._model.b, type="eq")
        problem = OptimizationProblem(loss_fun=lossfun, loss_grad=lossgrad,
                                      start=fvalue.initial_value,
                                      eqcon = equality_constraint,
                                      incon = cost_constraint,
                                      lb = fvalue.lower_bound(self._model.lb)
                                      )
        return problem

    @staticmethod
    def _negative(v):
        vminus = -v
        vneg = vminus.clip(min=0.)
        return vneg
