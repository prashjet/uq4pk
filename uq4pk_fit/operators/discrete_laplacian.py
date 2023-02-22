"""
Contains class 'DiscreteLaplacian'.
"""

import numpy as np
import scipy.ndimage as spim

from typing import Sequence

from .regop import RegularizationOperator


class DiscreteLaplacian(RegularizationOperator):
    """
    Implements the discrete Laplace operator for dim-dimensional input.
    """
    def __init__(self, shape: Sequence[int], mode="reflect", cval=0.0):
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
        """
        x = self._mat @ v
        return x

    def adj(self, v: np.ndarray) -> np.ndarray:
        return self._mat.T @ v

    def inv(self, w: np.ndarray) -> np.ndarray:
        """
        Compute inverse via least-squares method.
        """
        v = np.linalg.lstsq(self._mat, w)
        return v

    # PRIVATE

    def _compute_mat(self):
        basis = np.identity(self.dim)
        lap_list = []
        for column in basis.T:
            im = np.reshape(column, self._shape)
            lap = self._laplacian(im).flatten()
            lap_list.append(lap)
        l_mat = np.column_stack(lap_list)
        return l_mat

    def _laplacian(self, arr: np.ndarray) -> np.ndarray:
        """
        Applies the Laplacian to an N-dimensional array.
        """
        lap = spim.laplace(input=arr, mode=self._mode, cval=self._cval)
        return lap
