"""
Contains class 'DiscreteLaplacian'.
"""

import numpy as np
import cv2
import scipy.ndimage as spim

from typing import Sequence

from ..cgn import RegularizationOperator


class DiscreteLaplacian(RegularizationOperator):
    """
    Implements the discrete Laplace operator for dim-dimensional input.
    """
    def __init__(self, shape: Sequence[int], mode="reflect", cval=0.0):
        """

        :param shape: The shape of the input. E.g. (m, dim) for an image of shape (m, dim).
        :param mode: Determines how the boundaries are handled. For possible options, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html.
        :param cval: See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html.
        """
        self.name = "DiscreteLaplacian"
        self._shape = shape
        self._mode = mode
        self._cval = cval
        self._dim = 1
        for d in shape:
            self._dim *= d
        self._rdim = self.dim
        mat = self._compute_mat()
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        Computes discrete Laplacian of the flattened array v.
        :param v: Of shape (dim, ).
        :return: Of shape (dim, )
        """
        x = self._mat @ v
        return x

    def adj(self, v: np.ndarray) -> np.ndarray:
        """
        :param v:
        :return:
        """
        return self._mat.T @ v

    def inv(self, w: np.ndarray) -> np.ndarray:
        """
        Compute inverse via least-squares method (since I'm worried about stability).
        """
        v = np.linalg.lstsq(self._mat, w)
        return v

    # PRIVATE

    def _compute_mat(self):
        basis = np.identity(self.dim)
        l_list = []
        for column in basis.T:
            im = np.reshape(column, self._shape)
            l = self._laplacian(im).flatten()
            l_list.append(l)
        l_mat = np.column_stack(l_list)
        return l_mat

    def _laplacian(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the Laplacian to an N-dimensional array.

        :param arr: An N-dimensional image.
        :return: An N-dimensional image of the same shape as ``arr``.
        """
        lap = spim.laplace(input=arr, mode=self._mode, cval=self._cval)
        return lap
