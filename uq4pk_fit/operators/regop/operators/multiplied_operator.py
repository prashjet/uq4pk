"""
Contains class multiplied operator
"""

from copy import deepcopy
import numpy as np

from ..regularization_operator import RegularizationOperator


class MultipliedOperator(RegularizationOperator):
    """
    Implements a regularization operator that is created by right-multiplying a given regularization operator
    with an invertible matrix. That is, given a regularization operator :math`R` and a matrix :math:`Q`, the
    new regularization operator corresponds to :math:`R Q`.
    """
    def __init__(self, regop: RegularizationOperator, q: np.ndarray):
        """
        Parameters
        ---
        regop
            The regularization operator :math:`R`.
        q
            The matrix :math:`Q` by which the regularization operator is multiplied. It must have shape (dim,m),
            where dim = :code:`regop.dim`.
        """
        self._op = deepcopy(regop)
        self._q = q.copy()
        mat = regop._mat @ q
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray):
        u = self._q @ v
        return self._op.fwd(u)

    def adj(self, v: np.ndarray):
        # (RQ)^* = Q.T R^*
        return self._q.T @ self._op.adj(v)

    def inv(self, w: np.ndarray):
        """
        If RQ v = w then Q v = R^(-1) z.
        """
        qv = self._op.inv(w)
        v = np.linalg.solve(self._q, qv)
        return v
