
import numpy as np

from .downsampling import Downsampling
from .linear_interpolation import linear_interpolation1d


class Downsampling1D:
    """
    Given a vector of size n, downsamples it to size ceil(n / d), and upsamples by linear interpolation.
    """
    def __init__(self, n: int, d: int):
        """

        :param n: Size of the vector.
        :param d: Downsampling ratio.
        """
        self.dim = n
        self.rdim = np.ceil(n / d).astype(int)
        self._d = d
        # Create index set.
        indices = np.arange(0, n, d)
        self._indices = indices

    def indices(self) -> np.ndarray:
        return self._indices

    def reduce(self, i: int) -> int:
        """
        The representing index for i is simply i // d.
        :param i:
        """
        return i // self._d

    def downsample(self, x: np.ndarray):
        assert x.shape == (self.dim, )
        u = x[self._indices]
        assert u.shape == (self.rdim, )
        return u

    def enlarge(self, u: np.ndarray):
        """
        Enlarges by linear interpolation.
        :param u: (k, rdim)-array.
        :return: (k, dim)-array.
        """
        assert u.shape[1] == self.rdim
        # Compute x using linear interpolation.
        x_list = []
        for ui in u:
            x = linear_interpolation1d(ui, self._indices, self.dim)
            assert x.shape == (self.dim, )
            x_list.append(x)
        x = np.row_stack(x_list)
        assert x.shape[0] == u.shape[0]
        return x
