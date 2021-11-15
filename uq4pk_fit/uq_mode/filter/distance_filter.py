
import numpy as np
from typing import Union

from .matrix_filter import MatrixFilter


class DistanceFilter(MatrixFilter):
    """
    A distance filter is a filter that is determined by the distance function. The distance function is then
    translated into a weight via a weighting function.
    """
    def __init__(self, m: int, n: int, a: int, b: int, position: np.ndarray, weighting: callable,
                 scaling: Union[float, np.ndarray]):
        """
        :param m: Number of rows of the image.
        :param n: Number of columns of the image.
        :param a: Vertical width of the localization window.
        :param b: Horizontal width of the localization window.
        :param position: Position of the filter center.
        :param weighting: The weighting function.
        :param scaling: A factor with which the distance is multiplied. This is used in downsampling, so that
            the distance function still refers to the distances in the original image.
        """
        self._weighting = weighting
        self.a = a
        self.b = b
        self._scaling = scaling
        rows = 2 * a + 1
        cols = 2 * b + 1
        # initialize the weight matrix
        k = np.zeros([rows, cols])
        center = np.array([a, b])
        # set the weights
        for i in range(rows):
            for j in range(cols):
                pos_ij = np.array([i, j])
                d_ij = self._dist(pos_ij, center)
                k[i, j] = self._weighting(d_ij)
        MatrixFilter.__init__(self, m=m, n=n, position=position, mat=k, center=center, normalized=True)

    # PROTECTED

    def _dist(self, pos1, pos2):
        """
        Distance function.
        :param pos1: array_like, shape (2,), type int
        :param pos2: array_like, shape (2,), type int
        :return: float
            The distance between the two points.
        """
        d = np.linalg.norm((pos1 - pos2) / self._scaling)
        return d