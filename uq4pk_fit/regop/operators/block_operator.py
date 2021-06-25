"""
Contains class 'BlockOperator'
"""


import numpy as np
import scipy.linalg

from ..regularization_operator import RegularizationOperator



class BlockOperator(RegularizationOperator):
    """
    Given a list of regularization operators P1, ..., Pl, we form the block operator
    P = diag(P1, P2, ..., Pl).
    """
    def __init__(self, operator_list, dim_list, rdim_list):
        """
        :param operator_list: A list of Covroot objects
        :param n_list: A list of integers, corresponding to the input dimensions (n1, n2, ...)
        :param r_list: A second list of integers, corresponding to the output dimensions (r1, r2, ...)
        of the covroots. Its length should equal the length of regop_list
        """
        RegularizationOperator.__init__(self)
        assert len(operator_list) == len(dim_list)
        self._operators = operator_list
        self._n_split_positions = []
        self._r_split_positions = []
        i = 0
        j = 0
        # get all the positions in the vector where a new vector *starts*
        for n, r in zip(dim_list[:-1], rdim_list[:-1]):
            self._n_split_positions.append(i + r)
            self._r_split_positions.append(j + n)
            i += r
            j += n
        self.dim = sum(dim_list)
        self.rdim = sum(rdim_list)
        mat_list = []
        imat_list = []
        for op in operator_list:
            mat_list.append(op.mat)
            imat_list.append(op.imat)
        self.mat = scipy.linalg.block_diag(*mat_list)
        self.imat = scipy.linalg.block_diag(*imat_list)

    def fwd(self, v):
        """
        Splits
        :param v: a vector of size n_1 + n_2 + ...
        :return: a vector of size r_1 + r_2 + ...
        """
        v_list = np.split(v, self._r_split_positions, axis=0)
        res_list = []
        for op, vec in zip(self._operators, v_list):
            u = vec
            sol = op.fwd(u)
            res_list.append(sol)
        return np.concatenate(res_list)

    def inv(self, v):
        """
        Given a vector applies the block diagonal vector to it
        :param v: a vector of shape (r_1 + r_2 + ...,)
        :return: a vector of shape (n_1 + n_2 + ...,)
        """
        assert v.size == self.rdim
        v_list = np.split(v, self._n_split_positions, axis=0)
        res_list = []
        for op, u in zip(self._operators, v_list):
            res_list.append(op.inv(u))
        return np.concatenate(res_list, axis=0)

    def right(self, v):
        v_list = np.split(v, self._r_split_positions, axis=1)
        res_list = []
        for op, u in zip(self._operators, v_list):
            res_list.append(op.right(u))
        return np.concatenate(res_list, axis=1)