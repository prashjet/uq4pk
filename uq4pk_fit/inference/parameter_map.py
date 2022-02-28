"""
Contains class "ParameterMap".
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import List

from ..cgn import MultipliedOperator


class ParameterMap:
    """
    Handles switching from the parameter x (a List) to the parameters f and theta of the right dimension.
    """
    def __init__(self, dim_f, dim_theta):
        # make a mask for the fixed theta value
        self._dim_f = dim_f
        self._dim_theta = dim_theta
        self._index_fixed = np.full((self._dim_theta), False, dtype=bool)
        self._theta_values = np.zeros(self._dim_theta)

    @property
    def dims(self) -> List:
        """
        The dimensions of the parameter
        :return: list[int]
        """
        dims = [self._dim_f]
        if not self.theta_fixed:
            variable_dims = np.count_nonzero(self._index_fixed == False)
            dims.append(variable_dims)
        return dims

    @property
    def theta_fixed(self) -> bool:
        """"
        :returns: bool
            True if theta is completely fixed. Otherwise False.
        """
        is_fixed = np.all(self._index_fixed)
        return is_fixed

    @property
    def full_mask(self):
        """
        Returns the full mask for (f, theta_v)
        """
        mask_f = np.full((self._dim_f, ), True, dtype=bool)
        mask_theta = np.invert(self._index_fixed)
        mask = np.concatenate([mask_f, mask_theta], axis=0)
        return mask

    def fix_theta(self, indices: ArrayLike, values: ArrayLike):
        """
        Allows to fix parts of theta.
        """
        # If indices is list, turn into array
        indarr = np.array(indices)
        # Check that input makes sense.
        assert indarr.size == values.size
        assert set(indarr).issubset(set(np.arange(self._dim_theta)))
        self._index_fixed[indarr] = True
        self._theta_values[indarr] = values

    def x(self, f: ArrayLike, theta: ArrayLike) -> List:
        """
        Translates f and theta to x.
        :returns List: x
        """
        x_list = [f]
        if not self.theta_fixed:
            x1 = theta[~self._index_fixed]
            x_list.append(x1)
        return x_list

    def f_theta(self, x: List):
        """
        Translates x to f and theta.
        :param x:
        :returns: f, theta
        """
        assert len(x) >= 1
        f = x[0]
        theta = self._theta_values.copy()
        if not self.theta_fixed:
            assert len(x) >= 2
            theta[~self._index_fixed] = x[1]
        return f, theta

    def ci_f_theta(self, ci_x):
        """
        Translates credible interval for the concatenated vector to credible intervals for f and theta.
        :param ci_x: (dim,2) array, where dim = self.dim_x
        :return: ci_f, ci_theta
        """
        ci_f = ci_x[:self._dim_f, :]
        ci_theta = np.column_stack([self._theta_values, self._theta_values])
        if not self.theta_fixed:
            ci_theta[~self._index_fixed, :] = ci_x[self._dim_f:, :]
        return ci_f, ci_theta

    def p_x(self, p_f, p_theta) -> List:
        """
        Translates p_f and p_theta to p_x
        :return: List[ArrayLike]
        """
        p_list = [p_f]
        if not self.theta_fixed:
            # multiple p_theta with the indexing matrix
            p_vartheta = self.p_vartheta(p_theta)
            p_list.append(p_vartheta)
        return p_list

    def p_vartheta(self, p_theta):
        """
        Returns the regularization operator for the variable parts of theta.
        """
        # multiple p_theta with the indexing matrix
        idmat = np.eye(self._dim_theta)
        emb_mat = idmat[:, ~self._index_fixed]
        p_vartheta = MultipliedOperator(regop=p_theta, q=emb_mat)
        return p_vartheta

    @property
    def dim_x(self):
        """
        Dimension of the combined parameter vector.
        :return: int
        """
        dim = self._dim_f
        dim_theta = np.count_nonzero(self._index_fixed == 0)
        dim += dim_theta
        return dim
