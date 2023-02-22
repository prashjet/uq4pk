
import numpy as np

from .bessel1d import bessel1d


def bessel2d(image: np.ndarray, sigma: np.ndarray, truncate: float = 4.0, cval: float = 0):
    """
    Applies the discrete analog of the Gaussian filter to a given image.
    The filter is implemented as a sequence of two one-dimensional convolutions.

    Parameters
    ----------
    image : shape (m, n)
        The input image to the filter.
    sigma : shape (2, )
        Standard deviations of the kernel. First entry is the vertical, second entry the horizontal standard deviation.
    truncate :
        Truncate the kernel at this many standard deviations.

    Returns
    -------
    out : shape (m, n)
        The filtered image.
    """
    # Check input for consistency
    assert image.ndim == 2, "'image' must be two-dimensional array."
    assert np.all(sigma) > 0, "'sigma' must be strictly positive."

    # For both vertical and horizontal axis, convolve image with kernel.
    for axis, sig in zip(range(image.ndim), sigma):
        image = bessel1d(image, axis, sig, truncate, "reflect", cval)

    # return filtered image.
    return image