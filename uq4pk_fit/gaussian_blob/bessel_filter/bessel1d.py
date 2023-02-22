
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
    Note that the filter computations are only stable for sigma < ~25.
    To stabilize the Bessel filter, we use the Gaussian filter for all sigma > 15.
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