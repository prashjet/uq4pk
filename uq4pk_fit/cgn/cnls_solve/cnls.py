"""
Contains class CNLS.
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike

from ..regop import RegularizationOperator
from .cnls_constraint import CNLSConstraint, NullConstraint


class CNLS:
    """
    Contained class that manages the user-provided description of the
    constrained nonlinear least-squares problem:
    min_x (1/scale) * ( 0.5*||Q func(x) ||^2 + 0.5*||R(x-m)||^2 )
    s.t. Ax = b, Cx >= d, lb <= x <= ub
    The regularization term is optional.
    """
    def __init__(self, func: callable, jac: callable, q: RegularizationOperator, m: ArrayLike, r: RegularizationOperator,
                 eqcon: CNLSConstraint, incon: CNLSConstraint, lb: ArrayLike, ub: ArrayLike, scale: float):
        self._check_input(m, r, eqcon, incon, lb, ub)
        self.func = deepcopy(func)
        self.jac = deepcopy(jac)
        self.q = deepcopy(q)
        self.m = deepcopy(m)
        self.r = deepcopy(r)
        self.scale = scale
        self.dim = m.size
        self.a = deepcopy(eqcon.a)
        self.b = deepcopy(eqcon.b)
        self.c = deepcopy(incon.a)
        self.d = deepcopy(incon.b)
        self.lb = deepcopy(lb)
        self.ub = deepcopy(ub)
        self.equality_constrained = not isinstance(eqcon, NullConstraint)
        self.inequality_constrained = not isinstance(incon, NullConstraint)
        self.bound_constrained = np.isfinite(self.lb).any() or np.isfinite(self.ub).any()

    def satisfies_constraints(self, x, tol=1e-5):
        """
        Assert that given vector satisfies constraints up to a given tolerance.
        The error norm is the l^1-norm
        """
        constraint_error = 0.
        if self.equality_constrained:
            constraint_error += np.linalg.norm(self.a @ x - self.b)
        if self.inequality_constrained:
            constraint_error += np.linalg.norm((self.c @ x - self.d).clip(max=0.))
        if self.bound_constrained:
            constraint_error += np.linalg.norm((self.lb - x).clip(min=0.))
            constraint_error += np.linalg.norm((x - self.ub).clip(min=0.))
        if constraint_error <= tol:
            return True
        else:
            return False

    @staticmethod
    def _check_input(mean, regop, eqcon, incon, lb, ub):
        n = mean.size
        assert regop is None or regop.dim == n
        assert eqcon.dim == n
        assert incon.dim == n
        assert lb.size == n
        assert ub.size == n
