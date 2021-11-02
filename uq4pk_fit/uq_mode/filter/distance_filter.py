"""
Contains the abstract base class "DistanceFilter"
"""

import numpy as np

from .matrix_filter import MatrixFilter


class DistanceFilter(MatrixFilter):
    """
    A distance filter is a filter that is determined by the distance function. The distance function is then
    translated into a weight via a weighting function.
    """
    def __init__(self, m, n, a, b, position, weighting):
        """
        :param m: int
            Number of rows of the image.
        :param n: int
            Number of columns of the image.
        :param a: int
            Vertical width of the localization window.
        :param b: int
            Horizontal width of the localization window.
        :param c: int
            Vertical width of the localization window.
        :param d: int
            Vertical width of the
        :param weighting: callable
            The weighting function.
        """
        self._weighting = weighting
        self.a = a
        self.b = b
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
        d = np.linalg.norm(pos1 - pos2)
        return d