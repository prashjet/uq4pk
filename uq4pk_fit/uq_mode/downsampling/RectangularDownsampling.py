
import numpy as np
from typing import Tuple

from ..partition import rectangle_partition
from .downsampling import Downsampling


class RectangularDownsampling(Downsampling):
    """
    Performs downsampling for image-shaped parameters by partitioning the image into rectangles of shape (a, b),
    and then mapping each pixel in a given rectangle to the upper left corner pixel.
    """

    def __init__(self, shape: Tuple[int], a: int, b: int):
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

    def enlarge(self, u: np.ndarray):
        """

        :param u: Of shape (k, rdim).
        :return: Of shape (k, dim).
        """
        assert u.shape[1] == self.rdim
        k = u.shape[0]
        # Initialize enlarged stack.
        x = np.zeros((k, self.dim))
        # Now, simply loop over partition and assign back.
        for i in range(self.rdim):
            element_i = self._partition.element(i)
            x[:, element_i] = u[:, i].reshape(-1, 1)
        return x


