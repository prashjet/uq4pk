
import numpy as np

from ..evaluation import AffineEvaluationFunctional
from ..filter import LinearFilter


class FilterFunctional(AffineEvaluationFunctional):
    """
    Special case of :py:class:`AffineEvaluationFunctional` based on a linear filter.
    Given the filter with indices I and weight vector w, the corresponding affine evaluation functional is
    w = w
    phi(z) = w @ z
    x(z) = x_map + zeta_I z
    lb(z) = (lb - x_map)_I
    """

    def __init__(self, filter: LinearFilter, x_map: np.ndarray):
        self.w = filter.weights
        self.indices = filter.indices
        self.dim = x_map.size
        self.zdim = filter.size
        # Set up zeta matrix.
        zeta = np.zeros((self.dim, self.zdim))
        id_l = np.identity(self.zdim)
        zeta[self.indices, :] = id_l[:, :]
        # x(z)_I = z, x(z)_{~I} = x_map_{~I} ==> U = Zeta, v_I = 0, v_{~I} = x_map_{~I}.
        self.u = zeta
        self.v = x_map.copy()
        self.v[self.indices] = 0.
        self.z0 = x_map[self.indices]
        self.phidim = 1

    def phi(self, z: np.ndarray) -> np.ndarray:
        """
        For a filter, phi(z) = w @ z.
        """
        return np.array(self.w @ z).reshape((1, ))

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        """
        The lower bound on z is simply lb_I.
        """
        return lb[self.indices]
