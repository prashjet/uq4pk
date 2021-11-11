"""
Contains class "Parameter".
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike

from ..regop import RegularizationOperator, IdentityOperator, NullOperator

EPS = np.finfo(float).eps


class Parameter:
    """
    Container class for a user-defined parameter. Each parameter has an accompanying regularization term
    :math:`\\beta ||R(x-m)||_2^2`, which the user can modify by setting the corresponding attributes :py:attr:`beta`,
    :py:attr:`regop` and :py:attr:`mean`.
    Note that the default values are :math:`\\beta = 0`, :math:`m = 0` and :math:`R = I`.

    Furthermore, one can set a lower-bound constraint on the parameter by modifying the attribute :py:attr:`lb`.
    The default lower bound is :math:`-\\infty`.

    :ivar name: The name of the parameter.
    """
    def __init__(self, dim: int, name: str):
        self.name = name
        self._dim = dim
        self._beta, self._mean, self._regop, self._lb = self._default_values()
        self._rdim = self._regop.rdim

    # variable properties

    @property
    def beta(self) -> float:
        """
        The regularization parameter.
        """
        return self._beta

    @beta.setter
    def beta(self, value: float):
        try:
            self._beta = float(value)
        except:
            raise Exception("'beta' must be a float.")

    @property
    def mean(self) -> np.ndarray:
        """
        The prior mean in the regularization term.
        """
        return self._mean.copy()

    @mean.setter
    def mean(self, value: np.ndarray):
        if value.shape != (self._dim, ):
            raise Exception(f"'mean' must be of dimension {self._dim}.")
        else:
            self._mean = value

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @lb.setter
    def lb(self, value: np.ndarray):
        if value.shape != (self._dim, ):
            raise Exception(f"Lower bound must be an array of shape ({self._dim},)")
        else:
            self._lb = value

    @property
    def regop(self) -> RegularizationOperator:
        """
        The regularization operator for the parameter.
        """
        return self._regop

    @regop.setter
    def regop(self, value: RegularizationOperator):
        if value.dim != self._dim:
            raise Exception(f"'regop' must be of dimension {self._dim}.")
        else:
            self._regop = value
            self._rdim = value.rdim

    # constant properties

    @property
    def dim(self) -> int:
        """
        The dimension of the parameter.
        """
        return self._dim

    @property
    def rdim(self):
        """
        The dimension of the codomain in which the regularization :math:`R` maps. This means that if :math:`R` is an
        :math:`(r, n)`-matrix, then :py:attr:`rdim` will equal :math:`r`.
        """
        return self._rdim

    def _default_values(self):
        """
        Sets the default values.
        Default for mean is zero vector.
        Default for regop is Identity operator.
        Default for beta is 0.
        """
        default_mean = np.zeros((self._dim,))
        default_regop = IdentityOperator(dim=self._dim)
        default_beta = 0.
        default_lb = - np.inf * np.ones(self._dim)
        return default_beta, default_mean, default_regop, default_lb





