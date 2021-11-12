
import numpy as np

from ..evaluation import AffineEvaluationFunctional


class LCIEvaluationFunctional(AffineEvaluationFunctional):
    """
    The evaluation functional needed for computing a local credible interval wrt an index set I.
    """

    def __init__(self, index_set: np.ndarray, x_map: np.ndarray):
        self._x_map = x_map
        self._index_set = index_set
        self.dim = x_map.size
        self.zdim = index_set.size
        self.phidim = self.zdim
        self.w = np.ones((1, ))
        self.u = np.ones((self.dim, 1))
        self.v = x_map
        self.z0 = np.zeros((1, ))

    def phi(self, z: np.ndarray) -> np.ndarray:
        return self.x(z)

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        return max((lb - self._x_map)[self._index_set]) * np.ones((1, ))

