
import numpy as np
import scipy.optimize

from .constraint import Constraint, NullConstraint


class OptimizationProblem:
    """
    Represents general optimization problems of the form
    min_x loss(x) s.t. g(x) = 0, h(x) >= 0 and x >= lb.
    """
    def __init__(self, loss_fun, loss_grad, eqcon: Constraint, incon: Constraint, lb: np.ndarray=None):
        """
        """
        self.loss_fun = loss_fun
        self.loss_grad = loss_grad
        self.eqcon = eqcon
        self.incon = incon
        self.lb = lb
        self.constraints = self._set_constraints()
        self.bnds = self._set_bnds()

    def check_constraints(self, x: np.ndarray, tol: float):
        """
        Checks whether a given point satisfies all constraints.

        :param x:
        :param tol: The error tolerance. An error is only returned if the constraint violation is above `tol` in the
            l1-norm.
        :returns:
            - satisfied: True, if all constraints are satisfied up to the given tolerance. Otherwise False.
            - message: An error message that specifies which constraints are violated.
        """
        eqcon_error = self._l1norm(self.eqcon.fun(x))
        incon_error = self._l1norm(self.incon.fun(x).clip(max=0.))
        bound_error = self._l1norm((x - self.lb).clip(max=0.))
        eqcon_violated = eqcon_error > tol
        incon_violated = incon_error > tol
        bound_violated = bound_error > tol
        constraints_satisfied = not (eqcon_violated or incon_violated or bound_violated)
        message = "The following constraints have been violated: \n"
        if eqcon_violated:
            message += "- equality constraint \n"
        if incon_violated:
            message += "- inequality constraint \n"
        if bound_violated:
            message += "- bound constraint \n"
        return constraints_satisfied, message

    def _set_constraints(self):
        constraints_list = []
        if not isinstance(self.eqcon, NullConstraint):
            constraints_list.append(self.eqcon.as_dict())
        if not isinstance(self.incon, NullConstraint):
            constraints_list.append(self.incon.as_dict())
        if len(constraints_list) == 0:
            return None
        else:
            return tuple(constraints_list)

    def _set_bnds(self):
        if self.lb is None:
            bnds = None
        else:
            ub = np.inf * np.ones_like(self.lb)
            bnds = scipy.optimize.Bounds(lb=self.lb, ub=ub)
        return bnds

    @staticmethod
    def _l1norm(x):
        return np.sum(np.abs(x))

