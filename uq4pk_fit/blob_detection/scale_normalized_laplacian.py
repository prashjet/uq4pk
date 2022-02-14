
import numpy as np
import scipy.ndimage as spim
from typing import Sequence


def scale_normalized_laplacian(ssr: np.ndarray, scales: Sequence[float], mode: str="reflect") -> np.ndarray:
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
    assert ssr.shape[0] == len(scales)

    # For each scale h, compute h * Laplacian(ssr[i]).
    snl_list = []
    for i in range(len(scales)):
        h_i = scales[i]
        delta_f_i = spim.laplace(ssr[i], mode=mode)
        snl_i = h_i * delta_f_i
        snl_list.append(snl_i)
    snl = np.array(snl_list)

    # Check that the scale-normalized Laplacian has the same shape as the original scale-space rep.
    assert snl.shape == ssr.shape

    return snl

