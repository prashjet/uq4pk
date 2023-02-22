
import numpy as np
from scipy.ndimage import laplace, generic_laplace, correlate1d
from typing import Sequence

from ..scale_to_sigma import sigma_to_scale2d


def scale_normalized_laplacian(ssr: np.ndarray, sigmas: Sequence[np.ndarray]) \
        -> np.ndarray:
    """
    Computes the scale-normalized Laplacian of a given scale-space representation.

    Parameters
    ----------
    ssr : shape (k, m, n)
        A scale-space object. The number `k` corresponds to the number of scales, while `m` and `n` are the image
        dimensions in vertical and horizontal direction.
    sigmas :
        The list of sigma-values.

    Returns
    -------
    snl : shape (k, m, n)
        The scale-normalized Laplacian of `ssr`.
    """
    # Check the input.
    assert ssr.ndim > 1
    assert ssr.shape[0] == len(sigmas)

    # For each scale h, compute h * Laplacian(ssr[i]).
    snl_list = []
    for i in range(len(sigmas)):
        t_i = sigma_to_scale2d(sigmas[i])
        snl_i = t_i * laplace(input=ssr[i], mode="reflect")
        snl_list.append(snl_i)
    snl = np.array(snl_list)

    # Check that the scale-normalized Laplacian has the same shape as the original scale-space rep.
    assert snl.shape == ssr.shape

    return snl


def scaled_laplacian(input: np.ndarray, t: np.ndarray, cval=0.0) -> np.ndarray:
    """
    Applies the scaled Laplacian operator,
    Delta_norm F = (t_1 \partial_1^2 + t_2 \partial_2^2) F.

    The implementation is based on scipy's generic_laplace filter.
    """
    def derivative_scaled(input, axis, output, mode, cval):
        return t[axis] * correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
    return generic_laplace(input, derivative_scaled, mode="reflect", cval=cval)


