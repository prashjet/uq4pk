
from math import sqrt
import numpy as np
from scipy.ndimage import convolve1d, correlate1d
from typing import Literal

from .bessel_kernel import bessel_kernel

Mode = Literal["constant", "reflect", "wrap", "nearest", "mirror"]


def bessel1d(image: np.ndarray, axis: int = 0, sigma: float = 1., truncate: float = 4.0, mode: Mode = "reflect",
             cval: float = 0.):
    """
    Implements one-dimensional Bessel filter.

    :param image: (m, n) array. The input image to the filter.
    :param axis: Axis along which to perform the convolution.
    :param sigma: Standard deviation of the kernel.
    :param mode: How to handle values outside the image borders.
    :param truncate: Truncate the kernel at this many standard deviations.
    :returns: (m, n) array. The filtered image.
    """
    # Translate sigma to scale.
    t = sigma * sigma
    # Get truncated Bessel kernel.
    r_trunc = np.ceil(truncate * sigma).astype(int)
    g = bessel_kernel(t, r_trunc)
    # Convolve with image along given dimension.
    output = convolve1d(image, g, axis=axis, mode=mode, cval=cval)
    # Return image.
    return output