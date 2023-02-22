
from math import sqrt
import numpy as np

from typing import Union


class CLS:
    """
    Represents an inequality-constrained least-squares problem.
    min_x ||H x - _y||_2^2 / scale s.t. lb <= x <= ub
    """
    def __init__(self, h: np.ndarray, y: np.ndarray, lb: np.ndarray = None, ub: np.ndarray = None, scale: float = 1.):
        self._check_consistency(h, y, lb, ub)
        self.h = h / sqrt(scale)
        self.y = y / sqrt(scale)
        if lb is not None:
            self.lb = lb
        else:
            n = self.h.shape[1]
            self.lb = -np.inf * np.ones(n)
        if ub is not None:
            self.ub = ub
        else:
            n = self.h.shape[1]
            self.ub = np.inf * np.ones(n)
        self.bound_constrained = np.isfinite(self.lb).any() or np.isfinite(self.u).any()

    @staticmethod
    def _check_consistency(h: np.ndarray, y: np.ndarray, lb: Union[np.ndarray, None], ub: Union[np.ndarray, None]):
        m, n = h.shape
        assert y.shape == (m, )
        if lb is not None:
            assert lb.shape == (n, )
        if ub is not None:
            assert ub.shape == (n, )
