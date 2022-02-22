
import numpy as np

from .discretization import AdaptiveDiscretization, Discretization


class TrivialDiscretization(Discretization):
    """
    The trivial discretization which is just the finest discretization.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.dof = dim

    @property
    def u(self) -> np.ndarray:
        """
        Returns the matrix u.
        """
        return np.identity(self.dim)

    @property
    def v(self) -> np.ndarray:
        """
        Returns the vector v.
        """
        return np.zeros(self.dim)

    def map(self, z: np.ndarray) -> np.ndarray:
        """
        Computes x = U z + v. (flattened)
        """
        return z

    def translate_lower_bound(self, lb: np.ndarray) -> np.ndarray:
        """
        Translates a lower bound on x into a corresponding lower bound on z.
        """
        return lb


class TrivialAdaptiveDiscretization(AdaptiveDiscretization):

    def __init__(self, dim: int):
        self.dim = dim
        self.discretizations = [TrivialDiscretization(dim=dim)] * dim