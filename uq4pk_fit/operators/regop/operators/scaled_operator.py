
from math import sqrt
import numpy as np


from ..regularization_operator import RegularizationOperator


class ScaledOperator(RegularizationOperator):
    """
    Represents a scaled regularization operator :math:`R = \\alpha P`.
    """
    def __init__(self, regop: RegularizationOperator, alpha: float):
        self._sqrt_a = sqrt(alpha)
        self._inv_sqrt_a = 1 / self._sqrt_a
        self._p = regop
        mat = self._sqrt_a * self._p._mat
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray):
        return self._sqrt_a * self._p.fwd(v)

    def adj(self, v: np.ndarray):
        return self._sqrt_a * self._p.adj(v)

    def inv(self, w: np.ndarray) -> np.ndarray:
        return self._inv_sqrt_a * self._p.inv(w)
