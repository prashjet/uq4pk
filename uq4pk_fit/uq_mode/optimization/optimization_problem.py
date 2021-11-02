
import numpy as np
import scipy.optimize

from .constraint import Constraint, NullConstraint


class OptimizationProblem:
    """
    Represents general optimization problems of the form
    min_x loss(x) s.t. g(x) = 0, h(x) >= 0 and x >= lb.
    """
    def __init__(self, loss_fun, loss_grad, start, eqcon: Constraint, incon: Constraint, lb: np.ndarray=None):
        """
        """
        # check that start satisfies constraint
        self._check_start(start, eqcon, incon, lb)
        self.loss_fun = loss_fun
        self.loss_grad = loss_grad
        self.start = start
        self.eqcon = eqcon
        self.incon = incon
        if lb is not None: assert lb.size == start.size
        self.lb = lb
        self.constraints = self._set_constraints()
        self.bnds = self._set_bnds()

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

    def _check_start(self, start, eqcon, incon, lb):
        assert np.isclose(eqcon.fun(start), 0.).all()
        c0 = incon.fun(start)
        assert (incon.fun(start) >= 0).all()
        if lb is not None:
            assert np.all(start >= lb - 1e-10)

    def _set_bnds(self):
        if self.lb is None:
            bnds = None
        else:
            ub = np.inf * np.ones_like(self.lb)
            bnds = scipy.optimize.Bounds(lb=self.lb, ub=ub)
        return bnds

