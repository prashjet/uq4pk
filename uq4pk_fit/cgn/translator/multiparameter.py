
from collections import UserList
from copy import deepcopy
from typing import List

import numpy as np

from uq4pk_fit.cgn.regop import RegularizationOperator, scale_operator, make_block_operator
from uq4pk_fit.cgn.problem.parameter import Parameter


class MultiParameter(UserList):
    """
    Manages a list of parameters. Individual parameters can be accessed via []
    """
    def __init__(self, parameter_list: List[Parameter]):
        UserList.__init__(self, initlist=parameter_list)
        self._nparams = len(parameter_list)
        self._positions = []
        self._dim = 0    # overall dimension of all parameters combined
        self._rdim = 0    # overall r of all parameters combined
        mean_list = []
        lb_list = []
        ub_list = []
        for param in parameter_list:
            dim = param.dim
            rdim = param.rdim
            self._positions.append(self._dim)
            self._dim += dim
            self._rdim = rdim
            mean_list.append(param.mean)
            lb_list.append(param.lb)
            ub_list.append(param.ub)
        self._combined_mean = np.concatenate(mean_list)
        self._comined_regop = self._combine_regops()
        self._combined_lb = np.concatenate(lb_list)
        self._combined_ub = np.concatenate(ub_list)

    @property
    def dim(self) -> int:
        """
        The overall dimension.
        """
        return self._dim

    @property
    def mean(self) -> np.ndarray:
        """
        The combined mean.
        """
        return self._combined_mean

    @property
    def regop(self) -> RegularizationOperator:
        """
        The combined regularization operator.
        """
        return self._comined_regop

    @property
    def lb(self) -> np.ndarray:
        """
        The combined lower bound.
        """
        return self._combined_lb

    @property
    def ub(self) -> np.ndarray:
        """
        The combined upper bound.
        """
        return self._combined_ub

    @property
    def nparams(self) -> int:
        """
        The number of components of the multi-parameter.
        """
        return self._nparams

    @property
    def rdim(self) -> int:
        return deepcopy(self._rdim)

    def extract_x(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Given a vector of size ``self.dim``, return a tuple of length ``self.nparams`` of vectors.
        """
        assert x.size == self.dim
        pvalues = []
        d0 = 0
        for i in range(self._nparams):
            param = self[i]
            dpa = param.dim
            pvalues.append(x[d0:d0+dpa])
            d0 += dpa
        return pvalues

    def position(self, i) -> int:
        """
        Returns the starting index of the i-th component in the concatenated parameter vector.
        """
        return self._positions[i]

    def position_by_name(self, name: str) -> int:
        """
        Returns the starting index of the parameter with name ``name``.
        """
        for i in range(len(self)):
            if self[i].name == name:
                return self.position(i)
        raise RuntimeError(f"Parameter with name {name} not found.")

    def _combine_regops(self) -> RegularizationOperator:
        regops_list = []
        for param in self:
            scaled_regop = scale_operator(regop=param.regop, alpha=param.beta)
            regops_list.append(scaled_regop)
        op = make_block_operator(operator_list=regops_list)
        return op
