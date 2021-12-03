
import numpy as np
from typing import Sequence

from ..cgn import RegularizationOperator
from .discrete_laplacian import DiscreteLaplacian


class NormalizedLaplacian(RegularizationOperator):

    """
    Implements the scale-normalized Laplacian in two dimension:
        \\Delta_{x,h} f(h, x) = h \\Delta_x f(h, x).
    It is assumed that f is given as flattened 3-dimensional array, i.e. we obtain the correct 3-dimensional
    representation
    by executing ``np.reshape(f, (nscales, m, n))``, so that the first axis corresponds to scale, and the two other axes
    are the two image dimensions.
    """

    def __init__(self, m: int, n: int, scales: Sequence[float]):
        """
        :param m: Height of the image.
        :param n: Width of the image.
        :param scales: The scale discretization.
        """
        self._m = m
        self._n = n
        self._scales = np.array(scales)
        self._nscales = len(scales)
        self._dim = m * n * self._nscales
        self._rdim = m * n * self._nscales
        # We also need the two-dimensional Laplacian
        self._lap2d = DiscreteLaplacian(m=m, n=n).mat
        mat = self._compute_mat()
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        return self._mat @ v

    def adj(self, u: np.ndarray) -> np.ndarray:
        return self._mat.T @ u

    def _compute_mat(self) -> np.ndarray:
        """
        Computes the scale-Laplacian matrix.
        """
        basis = np.identity(self.dim)
        d_list = []
        for column in basis.T:
            d = self._evaluate_normalized_laplacian(column)
            d_list.append(d)
        d_mat = np.column_stack(d_list)
        return d_mat

    def _evaluate_normalized_laplacian(self, flat: np.ndarray):
        """
        Evaluates \\Delta_{x,h} f(x, h) = h^{1/2} \\Delta_{x} f(x, h).
        :param flat: The flattened scale-space representation.
        :return: An array of the same shape as ``flat``.
        """
        # Rescale the input.
        f = np.reshape(flat, (self._nscales, self._m, self._n))
        # Evaluate the Laplacian in 3 dimensions.
        h_delta_f = self._evaluate3d(f)
        # Return flattened vector
        out = h_delta_f.flatten()
        assert out.size == self._rdim
        return out

    def _evaluate3d(self, f: np.ndarray) -> np.ndarray:
        """
        :param f: Of shape (self._nscales, self._m, self._n)
        :return: Of shape (self._nscales, self._m, self._n)
        """
        assert f.shape == (self._nscales, self._m, self._n)
        # Apply the Laplacian along the spatial_dimensions.
        delta_f = self._d_xx(f)
        # Multiply with scale.
        h_delta_f = self._scales[:, np.newaxis, np.newaxis] * delta_f
        # Return
        assert h_delta_f.shape == (self._nscales, self._m, self._n)
        return h_delta_f

    def _d_xx(self, f: np.ndarray) -> np.ndarray:
        """
        Computes \\Delta_x f(h, x)

        :param f: Of shape (nscales, m, n).
        :return: Of shape (nscales, m, n).
        """
        # Flatten the two last axes of f
        f2d = np.reshape(f, (self._nscales, self._m * self._n))
        # Then apply the 2-dimensional Laplace operator to each slice.
        laplacian_f_2d = (self._lap2d @ f2d.T)
        # Transpose and reshape
        laplacian_f_3d = np.reshape(laplacian_f_2d.T, (self._nscales, self._m, self._n))
        return laplacian_f_3d
