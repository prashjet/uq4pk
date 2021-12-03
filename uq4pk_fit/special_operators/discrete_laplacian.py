"""
Contains class 'DiscreteLaplacian'.
"""

import numpy as np
import cv2

from ..cgn import RegularizationOperator


class SecondDerivative(RegularizationOperator):

    def __init__(self, n: int):
        self._dim = n
        self._rdim = n
        mat = self._compute_mat()
        RegularizationOperator.__init__(self, mat)

    def adj(self, w: np.ndarray) -> np.ndarray:
        return self._mat.T @ w

    def fwd(self, v: np.ndarray) -> np.ndarray:
        w = self._mat @ v
        return w

    def _compute_mat(self):
        basis = np.identity(self.dim)
        d_list = []
        for column in basis.T:
            d = self._evaluate_second_derivative(column)
            d_list.append(d)
        d_mat = np.column_stack(d_list)
        return d_mat

    def _evaluate_second_derivative(self, vec):
        vec_ext = np.append(vec, vec[-1])
        vec_ext = np.insert(vec_ext, 0, vec[0])
        vec_plus = np.roll(vec_ext, -1)[1:-1]
        vec_minus = np.roll(vec_ext, 1)[1:-1]
        second_derivative = vec_plus - 2 * vec + vec_minus
        return second_derivative


class DiscreteLaplacian(RegularizationOperator):
    """
    Implements the discrete gradient operator for an image of shape n_x, n_y
    """

    def __init__(self, m, n):
        self.name = "DiscreteLaplacian"
        self._m = m
        self._n = n
        self._dim = m * n
        self._rdim = self.dim
        self._kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        mat = self._compute_mat()
        RegularizationOperator.__init__(self, mat)

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    def fwd(self, v):
        """
        Computes discrete gradient of the flattened image v.
        :param v: ndarray
        :return: ndarray (v.size,)
        """
        x = self._mat @ v
        return x

    def adj(self, v):
        """
        Right-multiplies the inverse discrete gradient matrix to a vector or matrix.
        :param v: ndarray
        :return: ndarray of same shape as v
        """
        return self._mat.T @ v

    # PRIVATE

    def _compute_mat(self):
        basis = np.identity(self.dim)
        l_list = []
        for column in basis.T:
            im = np.reshape(column, (self.m, self.n))
            l = self._laplacian(im).flatten()
            l_list.append(l)
        l_mat = np.column_stack(l_list)
        return l_mat

    def _filter_image_old(self, im: np.ndarray):
        """
        Computes discrete gradient of image
        :param im: Of shape (m, n).
        :return: Of shape (m, n). The filtered image.
        """
        # zero pad the image
        padded_image = cv2.copyMakeBorder(src=im, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT,
                                          value=0)
        l = cv2.filter2D(padded_image, -1, self._kernel)
        # remove borders
        l = l[1:-1, 1:-1]
        return l

    def _laplacian(self, im: np.ndarray) -> np.ndarray:
        """
        Applies the Laplacian to a 2-dimensional image.

        :param im: A 2-dimensional image.
        :return: A 2-dimensional image of the same shape as ``im``-
        """
        lap = cv2.Laplacian(src=im, ddepth=-1, borderType=cv2.BORDER_REFLECT)
        #lap = cv2.Laplacian(src=im, ddepth=-1, borderType=cv2.BORDER_CONSTANT)
        return lap
