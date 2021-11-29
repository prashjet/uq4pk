

import numpy as np
from typing import Union
from skimage import feature


RTHRESH = 0.005     # The relative threshold at which features are identified.
OVERLAP = 0.9       # The allowed overlap for features. That is, features are merged if their overlap is higher than
                    # this value.


def blob_dog(image: np.ndarray, minscale: float, maxscale: float) -> Union[np.ndarray, None]:
    """
    Detects blobs in an image using the difference-of-Gaussians method.
    See https://en.wikipedia.org/wiki/Difference_of_Gaussians.

    :param image: The image as 2d-array.
    :param minscale: The minimal scale at which features should be detected.
    :param maxscale: The maximal scale at which features should be detected.
    :return: The detected features are returned as an array of shape (k, 3), where each row corresponds to a feature
        and is of the form (i, j, s), where (i, j) is the index in ``image`` at which the feature was identified
        and s is the detected scale of the feature. If no features are detected, then None is returned.
    """
    # Check input.
    _check_input_blob_dog(image, minscale, maxscale)
    # Set the threshold intensity for detected features. The intensity is a constant multiple of the maximum of
    # ``image``.
    thresh = RTHRESH * image.max()
    # Call scipy's blob dog to detect features. (The output already has the desired form.)
    features = feature.blob_dog(image=image, min_sigma=minscale, max_sigma=maxscale, threshold=thresh, overlap=OVERLAP)
    # If no features are detected, we return None.
    if features.size == 0:
        features = None
    # Return it.
    return features


def _check_input_blob_dog(image: np.ndarray, minscale: float, maxscale: float):
    """
    Checks that ``image`` is indeed a 2d-array and that minscale <= maxscale.
    """
    if image.ndim != 2:
        raise Exception("`image` must be a 2-dimensional array.")
    if minscale > maxscale:
        raise Exception("'minscale' must not be larger than 'maxscale'.")

