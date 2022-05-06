
import numpy as np

from ..evaluation import AffineEvaluationFunctional
from ..filter import LinearFilter


class FilterFunctional(AffineEvaluationFunctional):
    """
    Special case of :py:class:`AffineEvaluationFunctional` based on a linear filter.
    Given the filter with weight vector k,
    creates an affine evaluation functional for solving the optimization problem
        min k.T x   s.t. x >= lb,
    """
    def __init__(self, filter: LinearFilter, x_map: np.ndarray):
        assert filter.dim == x_map.size
        self.dim = x_map.size
        self.zdim = self.dim
        self.w = filter.weights
        self.z0 = x_map

    @property
    def u(self) -> np.ndarray:
        return np.identity(self.dim)

    @property
    def v(self) -> np.ndarray:
        return np.zeros(self.dim)

    def phi(self, z: np.ndarray) -> float:
        """
        For a filter, phi(z) = w @ x(z).
        """
        return self.w @ z

    def x(self, z: np.ndarray) -> np.ndarray:
        """

        :param z:
        :return: x
        """
        return z

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        """
        Translates lower bound on x into lower bound on z.
        """
        return lb
