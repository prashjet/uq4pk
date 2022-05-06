
import numpy as np

from ..discretization import Discretization
from ..evaluation import AffineEvaluationFunctional
from ..filter import LinearFilter


class AdaptedFilterFunctional(AffineEvaluationFunctional):
    """
    Special case of :py:class:`AffineEvaluationFunctional` based on a linear filter.
    Given the filter with weight vector k, additional weights w,  and a discretization x(z) = U z + v,
    creates an affine evaluation functional for solving the optimization problem
        min kappa.T x   s.t. x >= lb,
    where kappa = (k * w).
    But we have kappa.T x = kappa.T (U z + v) = kappa.T U z + kappa.T v = a.T z + b,
    with a = U.T kappa, b = kappa.T v, and furthermore x >= lb <=> z >= lb_z, where lb_z is given by the discretization.
    Hence, we can equivalently solve the reduced optimization problem
        min a.T z + b   s.t. z >= lb_z.
    """
    def __init__(self, filter: LinearFilter, discretization: Discretization, x_map: np.ndarray, weights: np.ndarray):
        assert filter.dim == discretization.dim == x_map.size
        self.dim = x_map.size
        self.zdim = discretization.dof

        if weights is None:
            kappa = filter.weights
        else:
            assert weights.shape == filter.weights.shape
            kappa = weights * filter.weights
            assert kappa.shape == filter.weights.shape
        self._a = discretization.u.T @ kappa
        self.w = self._a
        self._b = kappa @ discretization.v
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
