
from math import acos, sqrt, pi
import numpy as np
from typing import Union
from skimage import morphology

from .scale_space_representation import scale_space_representation
from .scale_normalized_laplacian import scale_normalized_laplacian


def detect_features(image: np.ndarray, sigma_min: float, sigma_max: float, num_sigma: int = 10, overlap: float = 0.5,
                    rthresh: float = 0.1, mode: str = "constant", ratio: float = 1.) -> Union[np.ndarray, None]:
    """
    Detects blobs in an image using the difference-of-Gaussians method.
    See https://en.wikipedia.org/wiki/Difference_of_Gaussians.

    :param image: The image as 2d-array.
    :param sigma_min: The minimal radius at which features should be detected.
    :param sigma_max: The maximal radius at which features should be detected.
    :param n_sigma: The number of intermediate radii between `r_min` and `r_max`. For example, if `r_min = 1`,
        `r_max = 10` and `n_r = 3`, then the Laplacian-of-Gaussian is evaluated for the radii 1, 2.5, 5, 7.5 and 10.
    :param rthresh: The relative threshold for detection of blobs. A blob is only detected if it corresponds to a
        scale-space minimum of the scale-normalized Laplacian that is below ``rthresh * log_stack.min()``, where
        ``log_stack`` is the stack of Laplacian-of-Gaussians.
    :param overlap: If two blobs have a relative overlap larger than this number, they are considered as one.
    :param mode: Determines how the image boundaries are handled.
    :param ratio: Determines width/height ratio in feature detection.
    :return: The detected features are returned as an array of shape (k, 3), where each row corresponds to a feature
        and is of the form (w, h, i, j), where (i, j) is the index in ``image`` at which the feature was identified
        and ``w`` and ``h`` are the width and height of the feature. If no features are detected, then None is returned.
    """
    # Check input.
    _check_input(image, sigma_min, sigma_max)
    # Set the threshold intensity for detected features. The intensity is a constant multiple of the maximum of
    # ``image``.

    # Discretize sigma
    r_step = (sigma_max - sigma_min) / (num_sigma + 1)
    sigmas = [sigma_min + i * r_step for i in range(num_sigma + 2)]

    # COMPUTE LOG-STACK
    scales = [0.5 * sigma ** 2 for sigma in sigmas]
    # Compute scale-space representation.
    ssr = scale_space_representation(image, scales, mode, ratio)
    # Evaluate scale-normalized Laplacian
    log_stack = scale_normalized_laplacian(ssr, scales, mode="reflect")

    # DETERMINE SCALE-SPACE BLOBS
    # Determine local scale-space minima
    local_minima = morphology.local_minima(image=log_stack, indices=True)
    local_minima = np.array(local_minima).T

    if local_minima.size == 0:
        blobs = None
    else:
        # Bring output in correct format.
        blob_list = []
        for b in local_minima:
            w = sigmas[b[0]]
            h = ratio * w
            snl = log_stack[b[0], b[1], b[2]]
            blob_list.append(np.array([w, h, b[1], b[2], snl]))
        blobs = np.array(blob_list)

        # Remove all features below threshold.
        blobs = threshold_local_minima(blobs, rthresh)

        # Remove overlap
        blobs = remove_overlap(blobs, overlap, ratio)

        # Remove snl-column
        blobs = blobs[:, :-1]

        if blobs.size == 0:
            blobs = None

    return blobs


def _check_input(image: np.ndarray, r_min: float, r_max: float):
    """
    Checks that ``image`` is indeed a 2d-array and that minscale <= maxscale.
    """
    if image.ndim != 2:
        raise Exception("`image` must be a 2-dimensional array.")
    if r_min > r_max:
        raise Exception("'r_min' must not be larger than 'r_max'.")


def threshold_local_minima(blobs: np.ndarray, rthresh: float):
    """
    Removes all local minima with ssr[local_minimum] > rthresh * ssr.min().

    :param local_minima: Of shape (k, 5), where each row corresponds to a blob and is of the form (w, h, i, j, snl).
    :param rthresh: The relative threshold.
    :return: All local minima that are above threshold. Returns None if no local minimum is above threshold.
    """
    athresh = blobs[:, -1].min() * rthresh
    good_indices = np.where(blobs[:, -1] <= athresh)[0]
    return blobs[good_indices]


def remove_overlap(features: np.ndarray, othresh: float, ratio: float):
    """
    Given a features, removes overlap. The feature with the smaller scale-space Laplacian "wins".

    :param features: Of shape (k, 5). Each row corresponds to a feature and is of the form (w, h, i, j, snl).
    :param othresh: The maximum allowed overlap between features.
    :param ratio: The height/width ratio.
    :return: The remaining features. Of shape (l, 5), where l <= k.
    """
    feature_arr = features.copy()
    # Sort features in order of increasing snl.
    increasing_snl = np.argsort(feature_arr[:, -1])
    feature_arr = feature_arr[increasing_snl]
    # Convert to list
    feature_list = list(feature_arr)

    # Go through all features.
    cleaned_features = []
    while len(feature_list) > 0:
        feature = feature_list.pop(0)
        cleaned_features.append(feature)
        # Go through all other features.
        keep_indices = []
        for i in range(len(feature_list)):
            candidate_i = feature_list[i]
            overlap = compute_overlap(feature[[0, 2, 3]], candidate_i[[0, 2, 3]], ratio)
            # Since the features are sorted in order of increasing SNL, `candidate` must be the weaker feature.
            if overlap < othresh:
                keep_indices.append(i)
        # Remove all features to be removed.
        feature_list = np.array(feature_list)
        feature_list = feature_list[keep_indices]
        feature_list = list(feature_list)

    return np.array(cleaned_features)


def compute_overlap(feature1: np.ndarray, feature2: np.ndarray, ratio: float = 1):
    """
    Computes relative overlap of two features (sigma1, i, j) and (sigma2, k, l).
    r = sqrt(2) * sigma1, s = sqrt(2) * sigma2
    The relative overlap is I / a,
    where a is the area of the smaller circle, and I is the size of the intersection, i.e.
    I = 0, if d > s + r,
    I = 1, if d <= max(r-s, s-r),
    I = r1^2 acos(d_1 / r2) - d_1 sqrt(r1^2 - d_1^2) + r^2 acos(d_2 / r2) - d_w sqrt(r1^2 - d_2^2), otherwise,
    where d_1 = (r1^2 - r2^2 + d^2) / (2 d), d_2 = d - d_1, d is the distance of the features, and
    See https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6 for a derivation.

    :param feature1: Of shape (3, ).
    :param feature2: Of shape (3, ).
    :param ratio: The height/width ratio.
    :return: The relative overlap, a number between 0 and 1.
    """
    # Check input for right format.
    assert feature1.shape == feature2.shape == (3, )
    # Compute the distance of the two circles.
    pos1 = feature1[1:]
    pos2 = feature2[1:]
    # Note that radius = 2 * sqrt(scale)
    s = sqrt(2) * feature1[0]
    r = sqrt(2) * feature2[0]
    assert s > 0 and r > 0
    r1 = max(s, r)
    r2 = min(s, r)
    # Compute distance, adjusted for width/height ratio.
    d = np.linalg.norm(( pos1 - pos2) / np.array([ratio, 1.]))
    # Compute relative overlap
    if d > s + r:
        relative_overlap = 0.
    elif d <= r1 - r2:
        relative_overlap = 1
    else:
        # Compute area of smaller circle.
        a = pi * r2 ** 2
        # Compute area of intersection
        d1 = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        d2 = d - d1
        i11 = r1 ** 2 * acos(min(d1 / r1, 1))
        i12 = d1 * sqrt(max(r1 ** 2 - d1 ** 2, 0))
        i21 = r2 ** 2 * acos(min(d2 / r2, 1))
        i22 = d2 * sqrt(max(r2 ** 2 - d2 ** 2, 0))
        intersection = i11 - i12 + i21 - i22
        # Relative overlap is intersection / a.
        relative_overlap = intersection / a
    # Return relative overlap
    assert 0 <= relative_overlap <= 1
    return relative_overlap