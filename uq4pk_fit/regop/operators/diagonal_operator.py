"""
Contains class 'DiagonalOperator'
"""

import numpy as np

from ..regularization_operator import RegularizationOperator



class DiagonalOperator(RegularizationOperator):
    """
    Implements the diagonal operator P(v) = s * v.
    """
    def __init__(self, dim, s):
        """
        :param s: float or ndarray of shape (n,). All entries must me larger than 0.
        If s is a float, this corresponds to a diagonal covariance matrix with diagonal entries
        equal to 1 / s^2.
        If s is a vector, this corresponds to a diagonal covariance matrix with diagonal entries
        given by 1 / s^2.
        """
        RegularizationOperator.__init__(self)
        self.dim = dim
        self.rdim = dim
        assert np.all(s > 0)
        self._s = s
        self._s_inv = 1 / s
        self.mat = s * np.identity(dim)
        self.imat = self._s_inv * np.identity(dim)

    def right(self, v):
        return self._s_inv * v

    def fwd(self, v):
        return self._s * v

    def inv(self, v):
        return self._s_inv * v
