"""
Contains class 'DiscreteLaplacian'.
"""

import numpy as np
import cv2

from ..cgn import RegularizationOperator


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
            l = self._evaluate_laplacian(im)
            l_list.append(l)
        l_mat = np.row_stack(l_list)
        return l_mat

    def _evaluate_laplacian(self, im):
        """
        Computes discrete gradient of image
        :param im:
        :return:
        """
        # zero pad the image
        padded_image = cv2.copyMakeBorder(src=im, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT,
                                          value=0)
        l = cv2.filter2D(padded_image, -1, self._kernel)
        # remove borders
        l = l[1:-1, 1:-1]
        return l.flatten()