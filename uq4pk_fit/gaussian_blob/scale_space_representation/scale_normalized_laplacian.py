
import numpy as np
from scipy.ndimage import laplace, generic_laplace, correlate1d
from typing import Sequence, Union

from ..scale_to_sigma import sigma_to_scale2d


def scale_normalized_laplacian(ssr: np.ndarray, sigmas: Sequence[np.ndarray], mode: str="reflect") \
        -> np.ndarray:
    """
    Computes the scale-normalized Laplacian of a given scale-space representation.

    :param ssr: The scale-space representation of a signal.
    :param scales: The scale-discretization. Must satisfy len(scales) = ssr.shape[0].
    :param mode: The way in which the boundaries are handled.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html.
    :return: An array of the same shape as ``ssr``.
    """
    # Check the input.
    assert ssr.ndim > 1
    assert ssr.shape[0] == len(sigmas)

    # For each scale h, compute h * Laplacian(ssr[i]).
    snl_list = []
    for i in range(len(sigmas)):
        t_i = sigma_to_scale2d(sigmas[i])
        snl_i = t_i * laplace(input=ssr[i], mode=mode)
        snl_list.append(snl_i)
    snl = np.array(snl_list)

    # Check that the scale-normalized Laplacian has the same shape as the original scale-space rep.
    assert snl.shape == ssr.shape

    return snl


def scaled_laplacian(input: np.ndarray, t: np.ndarray, mode="reflect", cval=0.0) -> np.ndarray:
    """
    Applies the scaled Laplacian operator,
    Delta_norm F = (t_1 \partial_1^2 + t_2 \partial_2^2) F.

    The implementation is based on scipy's generic_laplace filter.

    :param input: Of shape (m, n). The input image.
    :param t: Either float or two-dimensional vector.
    :return: Of shape (m, n).
    """
    def derivative_scaled(input, axis, output, mode, cval):
        return t[axis] * correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
    return generic_laplace(input, derivative_scaled, mode=mode, cval=cval)


