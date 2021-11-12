"""
Contains class "MatrixFilter"
"""

import numpy as np

from ..filter.geometry2d import rectangle_indices
from .linearfilter import LinearFilter


class MatrixFilter(LinearFilter):
    """
    Base class for two-dimensional filter that can be represented by a matrix kernel.
    """
    def __init__(self, m, n, position, mat, center, normalized=False):
        """
        :param m: int
        :param n: int
        :param pos: (int, int)
            The desired position of the matrix filter.
        :param mat: (k,l)-numpy array
        :param center: (2,) numpy array of ints
            Center indices for the matrix array, i.e. must denote the position that defines the "center" of the matrix.
        :param normalized: bool
            If True, the filter weights are always normalized so that they sum to 1.
        """
        self.m = m
        self.n = n
        k, l = mat.shape
        # Get the upper left and lower right corner of the rectangle that is created if we align the center of
        # "mat" with "position".
        upper_left = position - center
        lower_right = position + np.array([k-1, l-1]) - center
        # Get the indices that lie inside that rectangle, and the relative indices of the cut-lci_vs_fci.
        indices, relative_indices = rectangle_indices(m=m, n=n, upper_left=upper_left, lower_right=lower_right,
                                                      return_relative=True)
        mat_inside = mat.flatten()[relative_indices]
        # Create the filter from the rectangle indices and the weights.
        if normalized:
            mat_inside = mat_inside / np.sum(mat_inside)
        LinearFilter.__init__(self, indices=indices, weights=mat_inside)
