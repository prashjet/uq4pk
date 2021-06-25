
from math import sqrt

from ..regularization_operator import RegularizationOperator


class ScaledOperator(RegularizationOperator):
    """
    Given a RegularizationOperator P and a constant alpha, creates the scaled operator Ps = sqrt(alpha) * P
    """
    def __init__(self, alpha, p: RegularizationOperator):
        RegularizationOperator.__init__(self)
        assert alpha > 0, "alpha must be larger 0"
        self._sqrt_a = sqrt(alpha)
        self._inv_sqrt_a = 1 / self._sqrt_a
        self._p = p
        self.dim = p.dim
        self.rdim = p.rdim
        self.mat = self._sqrt_a * self._p.mat
        self.imat = self._inv_sqrt_a * self._p.imat

    def fwd(self, v):
        return self._sqrt_a * self._p.fwd(v)

    def right(self, v):
        return self._inv_sqrt_a * self._p.right(v)

    def inv(self, v):
        return self._inv_sqrt_a * self._p.inv(v)