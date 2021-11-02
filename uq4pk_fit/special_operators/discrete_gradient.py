"""
Contains class 'DiscreteGradient'.
"""

import cv2
import numpy as np

from ..cgn import RegularizationOperator


class DiscreteGradient(RegularizationOperator):
    """
    Implements the discrete gradient operator for an image of shape n1, n2
    """
    def __init__(self, m, n):
        self.name = "DiscreteGradient"
        self._m = m
        self._n = n
        self._dim = m * n
        self._rdim = 2 * self.dim
        self._xkernel, self._ykernel = self._compute_kernels()
        mat = self._compute_mat()
        RegularizationOperator.__init__(self, mat)

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    def adj(self, w: np.ndarray) -> np.ndarray:
        return self._mat.T @ w

    def fwd(self, v: np.ndarray) -> np.ndarray:
        w = self._mat @ v
        return w

    # PRIVATE

    @staticmethod
    def _compute_kernels():
        xkernel = np.zeros((3, 3))
        xkernel[1, 1] = -1
        xkernel[2, 1] = 1
        ykernel = np.zeros((3, 3))
        ykernel[1, 1] = -1
        ykernel[1, 2] = 1
        return xkernel, ykernel

    def _compute_mat(self):
        basis = np.identity(self.dim)
        d_list = []
        for column in basis.T:
            im = np.reshape(column, (self.m, self.n))
            d = self._evaluate_gradient(im)
            d_list.append(d)
        d_mat = np.row_stack(d_list)
        return d_mat

    def _evaluate_gradient(self, im):
        """
        Computes discrete gradient of image
        :param im:
        :return:
        """
        # zero pad the image
        padded_image = cv2.copyMakeBorder(src=im, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT,
                                          value=0.)
        dx = cv2.filter2D(padded_image, -1, self._xkernel)
        dy = cv2.filter2D(padded_image, -1, self._ykernel)
        # remove borders
        dx = dx[1:-1, 1:-1]
        dy = dy[1:-1, 1:-1]
        return np.row_stack((dx.flatten(), dy.flatten()))
