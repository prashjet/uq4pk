"""
Contains class multiplied operator
"""

import numpy as np

from ..regularization_operator import RegularizationOperator


class MultipliedOperator(RegularizationOperator):
    """
    Implements a regularization operator that is created by left-multiplying a given regularization operator
    with a matrix. That is, given a regularization operator P and a matrix Q, the new regularization operator
    corresponds to PQ.
    """
    def __init__(self, regop: RegularizationOperator, q):
        RegularizationOperator.__init__(self)
        self.dim = q.shape[1]
        self.rdim = regop.rdim
        self._op = regop
        self._q = q
        self.mat = regop.mat @ q
        self.imat = np.linalg.pinv(q) @ regop.imat

    def fwd(self, v):
        u = self._q @ v
        return self._op.fwd(u)

    def inv(self, w):
        return self.imat @ w

    def right(self, v):
        return v @ self.imat
