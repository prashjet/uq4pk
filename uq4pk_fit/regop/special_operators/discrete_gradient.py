"""
Contains class 'DiscreteGradient'.
"""

import cv2
import numpy as np

from ..regularization_operator import RegularizationOperator



class DiscreteGradient(RegularizationOperator):
    """
    Implements the discrete gradient operator for an image of shape n1, n2
    """
    def __init__(self, n1, n2):
        RegularizationOperator.__init__(self)
        self.name = "DiscreteGradient"
        #print("Initializing discrete gradient...")
        self.n1 = n1
        self.n2 = n2
        self.dim = n1 * n2
        self.rdim = 2 * self.dim
        self._xkernel, self._ykernel = self._compute_kernels()
        #print("Assembling matrix")
        self.mat = self._compute_mat()
        #print("Inverting matrix")
        self.imat = np.linalg.pinv(self.mat)
        #print("Initialization done.")

    def inv(self, v):
        """
        Computes *inverse* discrete gradient of the flattened image v.
        :param v: ndarray
        :return: ndarray (v.size,)
        """
        x = self.imat @ v
        return x

    def right(self, v):
        """
        Right-multiplies the inverse discrete gradient matrix to a vector or matrix.
        :param v: ndarray
        :return: ndarray of same shape as v
        """
        return v @ self.imat

    def fwd(self, v):
        """
        Computes discrete gradient of v
        :param v: ndarray
        :return: ndarray of same shape as v
        """
        w = self.mat @ v
        return w

    # PRIVATE

    def _compute_kernels(self):
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
            im = np.reshape(column, (self.n1, self.n2))
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
        padded_image = cv2.copyMakeBorder(src=im, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=0.)
        dx = cv2.filter2D(padded_image, -1, self._xkernel)
        dy = cv2.filter2D(padded_image, -1, self._ykernel)
        # remove borders
        dx = dx[1:-1, 1:-1]
        dy = dy[1:-1, 1:-1]
        return np.row_stack((dx.flatten(), dy.flatten()))