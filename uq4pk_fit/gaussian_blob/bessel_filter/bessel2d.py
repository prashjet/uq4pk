
import numpy as np
from scipy.ndimage._ni_support import _normalize_sequence
from typing import Tuple, Union

from .bessel1d import bessel1d, Mode


def bessel2d(image: np.ndarray, sigma: Union[float, Tuple[float]] = 1., truncate: float = 4.0,  mode: Mode = "reflect",
             cval: float = 0):
    """
    Applies the discrete analog of the Gaussian filter to a given image.
    The filter is implemented as a sequence of two one-dimensional convolutions.

    :param image: (m, n) array. The input image to the filter.
    :param sigma: Standard deviation of the kernel.
    :param mode: How to handle values outside the image borders.
    :param truncate: Truncate the kernel at this many standard deviations.
    :returns: (m, n) array. The filtered image.
    """
    # Check input for consistency
    assert image.ndim == 2, "'image' must be two-dimensional array."
    assert np.all(np.asarray(sigma)) > 0, "'sigma' must be strictly positive."
    assert mode in ["constant", "reflect", "wrap", "nearest", "mirror"], "Unknown mode."

    # For both vertical and horizontal axis, convolve image with kernel.
    sigma = _normalize_sequence(sigma, image.ndim)
    for axis, sig in zip(range(image.ndim), sigma):
        image = bessel1d(image, axis, sig, truncate, mode, cval)

    # return filtered image.
    return image