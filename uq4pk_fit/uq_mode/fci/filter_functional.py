
import numpy as np

from ..discretization import Discretization
from ..evaluation import AffineEvaluationFunctional
from ..filter import LinearFilter


class FilterFunctional(AffineEvaluationFunctional):
    """
    Special case of :py:class:`AffineEvaluationFunctional` based on a linear filter.
    Given the filter with indices I and weight vector w, the corresponding affine evaluation functional is
    w = w
    phi(z) = a @ z + b
    x(z) = U z + v
    lb(z) = (lb - x_map)_I
    """
    def __init__(self, filter: LinearFilter, discretization: Discretization, x_map: np.ndarray):
        assert filter.dim == discretization.dim == x_map.size
        self.dim = x_map.size
        self.zdim = discretization.dof
        self._a = discretization.u.T @ filter.weights
        self.w = self._a
        self._b = filter.weights @ discretization.v
        self._discretization = discretization
        self.z0 = discretization.translate_lower_bound(x_map)

    @property
    def u(self) -> np.ndarray:
        return self._discretization.u

    @property
    def v(self) -> np.ndarray:
        return self._discretization.v

    def phi(self, z: np.ndarray) -> float:
        """
        For a filter, phi(z) = w @ x(z).
        """
        return self._a @ z + self._b

    def x(self, z: np.ndarray) -> np.ndarray:
        """

        :param z:
        :return: x
        """
        return self._discretization.map(z)

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        """
        Translates lower bound on x into lower bound on z.
        """
        return self._discretization.translate_lower_bound(lb)
