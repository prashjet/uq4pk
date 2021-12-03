
import numpy as np
from typing import Sequence

from ..cgn import RegularizationOperator
from .gradient import mygradient


class DiscreteGradient(RegularizationOperator):
    """
    Implements the discrete gradient operator for an image of shape n1, n2
    """
    def __init__(self, shape: Sequence[int]):
        self.name = "DiscreteGradient"
        self._shape = shape
        self._dim = np.prod(np.array(shape))
        self._rdim = len(shape) * self._dim
        mat = self._compute_mat()
        RegularizationOperator.__init__(self, mat)

    @property
    def shape(self):
        return self._shape

    def adj(self, w: np.ndarray) -> np.ndarray:
        return self._mat.T @ w

    def fwd(self, v: np.ndarray) -> np.ndarray:
        w = self._mat @ v
        return w

    # PRIVATE

    def _compute_mat(self):
        basis = np.identity(self.dim)
        d_list = []
        for column in basis.T:
            arr = np.reshape(column, self._shape)
            d = self._evaluate_gradient(arr)
            d_list.append(d)
        d_mat = np.column_stack(d_list)
        return d_mat

    def _evaluate_gradient(self, arr: np.ndarray) -> np.ndarray:
        """
        Computes discrete gradient of an N-dimensional array.

        :param arr: The array.
        :return: A vector of size np.prod(np.array(arr.shape)). For example, if ``arr`` has shape (m, n, l), then
            the returned gradient is a vector of size m * n * l.
        """
        gradients = mygradient(arr)
        flattened_gradient = [grad.flatten() for grad in gradients]
        gradient = np.concatenate(flattened_gradient)
        return gradient