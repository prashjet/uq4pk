
from copy import deepcopy
import numpy as np


class RegularizationOperator:
    """
    Abstract base class for regularization operators.
    Each child of RegularizationOperator must implement the methods `fwd` and `adj`, which give the forward and the
    adjoint action of the regularization operator.
    """
    def __init__(self, mat: np.ndarray):
        if mat.ndim != 2:
            raise ValueError
        self._mat = deepcopy(mat)
        self._dim = mat.shape[1]
        self._rdim = mat.shape[0]

    @property
    def dim(self) -> int:
        """
        The dimension of the domain of the regularization operator.
        """
        return deepcopy(self._dim)

    @property
    def rdim(self) -> int:
        """
        The dimension of the codomain of the regularization operator.
        """
        return deepcopy(self._rdim)

    @property
    def mat(self) -> np.ndarray:
        """
        The matrix representation of the regularization operator, a matrix of shape (r,dim).
        """
        return deepcopy(self._mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        Evaluates the forward action of the regularization operator.
        """
        raise NotImplementedError

    def adj(self, w: np.ndarray) -> np.ndarray:
        """
        Evaluates the adjoint action of the regularization operator.
        """
        raise NotImplementedError

    def inv(self, w: np.ndarray) -> np.ndarray:
        """
        Returns the (pseudo-) inverse of the regularization operator, i.e. the solution v of R v = w.
        """
        raise NotImplementedError
