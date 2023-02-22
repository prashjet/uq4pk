
import numpy as np
from typing import Sequence

from ..bessel_filter import bessel2d


def scale_space_representation(image: np.ndarray, sigmas: Sequence[np.ndarray])\
        -> np.ndarray:
    """
    Computes the discrete Gaussian scale-space representation of a given image.

    Parameters
    ----------
    image : shape (m, n)
    sigmas :
        The list of sigma-values.

    Returns
    -------
    ssr : shape (k, m, n)
        Returns the scale-space representation, where `k` is the number of entries in `sigmas`.
    """
    scaled_images = []
    for sigma in sigmas:
        scaled_image = bessel2d(image, sigma=sigma)
        scaled_images.append(scaled_image)
    ssr = np.array(scaled_images)
    return ssr
