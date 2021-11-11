
import numpy as np
from typing import Literal, Union


class CNLSConstraint:

    _a: Union[np.ndarray, None]
    _b: Union[np.ndarray, None]
    _dim: int
    _cdim: int

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def cdim(self) -> int:
        return self._cdim


class ConcreteConstraint(CNLSConstraint):

    def __init__(self, dim: int, a: np.ndarray, b: np.ndarray):
        self._check_input(dim, a, b)
        self._dim = dim
        self._a = a
        self._b = b
        self._cdim = a.shape[0]

    def _check_input(self, dim: int, a: np.ndarray, b: np.ndarray):
        assert dim > 0
        assert a.shape[1] == dim
        assert b.shape == (a.shape[0], )


class NullConstraint(CNLSConstraint):

    def __init__(self, dim: int):
        assert dim > 0
        self._dim = dim
        self._a = None
        self._b = None
        self._cdim = 0


