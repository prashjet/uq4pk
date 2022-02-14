
import numpy as np
from typing import Sequence
from skimage import feature


def determinant_of_hessian(ssr: np.ndarray, scales: Sequence[float]) -> np.ndarray:
    """
    Computes the scale-normalized Determinant-of-Hessian operator,

    .. math::
        \\det H_{norm} L(x, t) = t^2 (\\partial_{xx} \\partial_{yy} - \\partial_{xy}^2) L(x, t).

    :param ssr: The 3-dimensional scale-space representation :math:`L=L(x,t)` as array of shape (k, m, n),
        where k is the number of scales.
    :param scales: The scale-discretization. Must satisfy :code:`len(scales) == ssr.shape[0]`.
    :return: An array of the same shape as ``ssr``.
    """
    # Check the input
    assert ssr.ndim == 3
    assert len(scales) == ssr.shape[0]

    # For each scale t, compute determinant-of-hessian.
    doh_list = []
    nscales = len(scales)
    im_shape = ssr[0].shape
    for i in range(nscales):
        l_t = ssr[i]
        # Compute determinant-of-Hessian
        doh_t = doh(l_t)
        # Scale-normalize
        t = scales[i]
        doh_t = (t ** 2) * doh_t
        # Store in list
        doh_list.append(doh_t)

    # Assemble in a single array.
    doh_arr = np.array(doh_list)

    # The Determinant-of-Hessian should have the same dimension as the original array.
    assert doh_arr.shape == ssr.shape

    return doh_arr


def doh(im: np.ndarray) -> np.ndarray:
    """
    Computes the determinant of Hessian of an image.

    :param im: 2-dimensional numpy array.
    :return: Image of same shape.
    """
    hess = image_hessian(im)

    det_hess = hess[0, 0] * hess[1, 1] - hess[0, 1] * hess[1, 0]

    return det_hess



def image_hessian(im: np.ndarray) -> np.ndarray:
    """
    Computes image Hessian.

    :param im: Of shape (m, n).
    :return: Of shape (2, 2, m, n).
    """
    gradients = np.gradient(im)
    hessian_rows = []
    for gradient in gradients:
        second_derivatives = np.gradient(gradient)
        row = np.asarray(second_derivatives)
        hessian_rows.append(row)

    hessian = np.array(hessian_rows)
    return hessian

