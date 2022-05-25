
import numpy as np
from typing import Tuple

from ..partition import rectangle_partition
from .downsampling import Downsampling
from .linear_interpolation import linear_interpolation


class RectangularDownsampling(Downsampling):
    """
    Performs downsampling for image-shaped parameters by partitioning the image into rectangles of shape (a, b),
    and then mapping each pixel in a given rectangle to the upper left corner pixel.
    """

    def __init__(self, shape: Tuple[int, int], a: int, b: int):
        """

        :param shape: The shape of the image.
        :param a: Maximal vertical length of the rectangles.
        :param b: Maximal horizontal length of the rectangles.
        """
        assert len(shape) == 2
        self.shape = shape
        self.dim = shape[0] * shape[1]
        self._a = a
        self._b = b
        m, n = shape
        self._rshape = (np.ceil(m / a).astype(int), np.ceil(n / b).astype(int))
        # Create rectangle partition
        self._partition = rectangle_partition(m=shape[0], n=shape[1], a=a, b=b)
        self.rdim = self._partition.size
        # Now, determine all the corner indices for the partition elements, which make up the index set I.
        self._indices = np.array([element[0] for element in self._partition.get_element_list()])

    def indices(self) -> np.ndarray:
        return self._indices

    def reduce(self, i: int) -> int:
        element_index = self._partition.in_which_element(i)
        j = self._indices[element_index]
        return j

    def downsample(self, x: np.ndarray):
        assert x.shape == (self.dim, )
        u = x[self._indices]
        assert u.shape == (self.rdim, )
        return u

    def enlarge(self, u: np.ndarray):
        """
        Enlarges using linear interpolation.

        :param u: Of shape (k, rdim).
        :return: Of shape (k, dim).
        """
        assert u.shape[1] == self.rdim
        # Reshape
        v = np.reshape(u, (-1, self._rshape[0], self._rshape[1]))
        # Compute x using linear interpolation.
        x_list = []
        for v_i in v:
            x_i = linear_interpolation(image=v_i, shape=self.shape, a=self._a, b=self._b).flatten()
            x_list.append(x_i)
        x = np.row_stack(x_list)
        return x


