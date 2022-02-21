
import numpy as np
from skimage import filters
from typing import Sequence, Union


def scale_space_representation(image: np.ndarray, scales: Sequence[Union[float, np.ndarray]], mode: str="constant")\
        -> np.ndarray:
    """
    Computes the Gaussian scale-space representation of a given image. That is, for an image f(x, y), it computes
    L(h, x, y), where L(h, x, y) = k_h * f(x, y), where k_h is the Gaussian kernel with scale parameter h.

    :param image: 2-dimensional image.
    :param scales: The scales.
    :param mode: Determines how the image is padded at the boundaries. See scikit.image.filters.gaussian.
    :param ratio: Height / width ratio.
    :return: Of shape (len(scales), m, n).
    """
    scaled_images = []
    for scale in scales:
        # Note that the scale parameter h corresponds to standard deviation sigma = sqrt(2 * h).
        sigma = np.sqrt(2 * scale)
        scaled_image = filters.gaussian(image, mode=mode, sigma=sigma)
        scaled_images.append(scaled_image)
    ssr = np.array(scaled_images)
    return ssr