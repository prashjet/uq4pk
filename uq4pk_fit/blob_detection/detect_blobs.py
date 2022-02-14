
import copy
import numpy as np
from typing import List, Sequence
from skimage import morphology
import shapely.affinity as aff
import shapely.geometry as geom

from .gaussian_blob import GaussianBlob
from .scale_space_representation import scale_space_representation
from .scale_normalized_laplacian import scale_normalized_laplacian


def detect_blobs(image: np.ndarray, sigma_min: float, sigma_max: float, num_sigma: int = 10, max_overlap: float = 0.5,
                 rthresh: float = 0.01, mode: str = "constant", ratio: float = 1.) -> List[GaussianBlob]:
    """
    Detects blobs in an image using the difference-of-Gaussians method.
    See https://en.wikipedia.org/wiki/Difference_of_Gaussians.

    :param image: The image as 2d-array.
    :param sigma_min: The minimal sigma at which blobs should be detected.
    :param sigma_max: The maximal sigma at which features should be detected.
    :param num_sigma: The number of intermediate sigma values between `sigma_min` and `sigma_max`. For example, if
        `sigma_min = 1`, `sigma_max = 15` and `num_sigma = 3`, then the Laplacian-of-Gaussian is evaluated for the
        sigma values 1, 2.5, 5, 7.5 and 10.
    :param rthresh: The relative threshold for detection of blobs. A blob is only detected if it corresponds to a
        scale-space minimum of the scale-normalized Laplacian that is below ``rthresh * log_stack.min()``, where
        ``log_stack`` is the stack of Laplacian-of-Gaussians.
    :param max_overlap: If two blobs have a relative overlap larger than this number, they are considered as one.
    :param mode: Determines how the image boundaries are handled.
    :param ratio: Determines the width/height ratio (y / x) of the ellipses.
    :return: The detected blobs are returned as an array of shape (k, 5), where each row corresponds to a feature
        and is of the form (x, y, sigma_x, sigma_y, ssl), where (x, y) is the position of the blob
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
    blobs = stack_to_blobs(scale_stack=log_stack, sigmas=sigmas, ratio=ratio, rthresh=rthresh, max_overlap=max_overlap)

    return blobs


def _check_input(image: np.ndarray, r_min: float, r_max: float):
    """
    Checks that ``image`` is indeed a 2d-array and that minscale <= maxscale.
    """
    if image.ndim != 2:
        raise Exception("`image` must be a 2-dimensional array.")
    if r_min > r_max:
        raise Exception("'r_min' must not be larger than 'r_max'.")


def threshold_local_minima(blobs: Sequence[GaussianBlob], thresh: float):
    """
    Removes all local minima with ssr[local_minimum] > rthresh * ssr.min().

    :param blobs: List of detected blobs.
    :param thresh: The absolute log-threshold below which a blob must lie.
    :return: The list of blobs with a log-value below the treshold.
    """
    blobs_below_tresh = [blob for blob in blobs if blob.log < thresh]
    return blobs_below_tresh


def remove_overlap(blobs: List[GaussianBlob], max_overlap: float):
    """
    Given a list of blobs, removes overlap. The feature with the smaller scale-space Laplacian "wins".

    :param blobs: Of shape (k, 5). Each row corresponds to a feature and is of the form (w, h, i, j, snl).
    :param max_overlap: The maximum allowed overlap between features.
    :return: The remaining features. Of shape (l, 5), where l <= k.
    """
    # Sort features in order of increasing log.
    blobs_increasing_log = best_blob_first(blobs)

    # Go through all blobs.
    cleaned_blobs = []
    while len(blobs_increasing_log) > 0:
        blob = blobs_increasing_log.pop(0)
        cleaned_blobs.append(blob)
        keep_list = []
        # Go through all other blobs.
        for candidate in blobs_increasing_log:
            overlap = compute_overlap(blob, candidate)
            # Since the features are sorted in order of increasing LOG, `candidate` must be the weaker blob.
            if overlap < max_overlap:
                keep_list.append(candidate)
        # Remove all blobs to be removed.
        blobs_increasing_log = keep_list

    return cleaned_blobs


def compute_overlap(blob1: GaussianBlob, blob2: GaussianBlob) -> float:
    """
    Computes the relative overlap of two blobs using shapely, i.e.

    .. math::
        o_r = \\frac{A_{intersection}}{\\min(A_1, A_2)}.

    The implementation uses shapely (https://pypi.org/project/Shapely/).

    :param blob1:
    :param blob2:
    :return: The relative overlap, a number between 0 and 1.
    """
    # Create shapely.ellipse objects
    ell1 = _create_ellipse(blob1)
    ell2 = _create_ellipse(blob2)

    # Compute areas of the two ellipses.
    a1 = ell1.area
    a2 = ell2.area
    # Compute intersection area.
    a_intersection = ell1.intersection(ell2).area

    # Compute relative overlap.
    relative_overlap = a_intersection / min(a1, a2)

    # Return relative overlap
    assert 0. <= relative_overlap <= 1.
    return relative_overlap


def _create_ellipse(blob: GaussianBlob):
    """
    Creates a shapely-ellipse object from a Gaussian blob.

    :param blob:
    :return: A shapely ellipse.
    """
    circ = geom.Point(blob.position).buffer(1)
    ellipse = aff.scale(circ, 0.5 * blob.width, 0.5 * blob.height)
    rotated_ellipse = aff.rotate(ellipse, angle=blob.angle)
    return rotated_ellipse


def stack_to_blobs(scale_stack: np.ndarray, sigmas: Sequence[float], ratio: float, rthresh: float, max_overlap: float)\
        -> List[GaussianBlob]:
    """
    Given a scale-space stack, detects blobs as scale-space minima.

    :param scale_stack:
    :param sigmas:
    :param ratio:
    :param rthresh:
    :param max_overlap:
    :return:
    """
    # DETERMINE SCALE-SPACE BLOBS
    # Determine local scale-space minima
    local_minima = morphology.local_minima(image=scale_stack, indices=True)
    local_minima = np.array(local_minima).T

    if local_minima.size == 0:
        blobs = []
    else:
        # Bring output in correct format.
        blobs = []
        for b in local_minima:
            sigma_y = sigmas[b[0]]
            sigma_x = ratio * sigma_y
            sslaplacian = scale_stack[b[0], b[1], b[2]]
            blob = GaussianBlob(x=b[2], y=b[1], sigma_x=sigma_x, sigma_y=sigma_y, log=sslaplacian)
            blobs.append(blob)

        # Remove all features below threshold.
        athresh = scale_stack.min() * rthresh
        blobs = threshold_local_minima(blobs, athresh)

        # Remove overlap
        blobs = remove_overlap(blobs, max_overlap)

    return blobs


def best_blob_first(blobs: List[GaussianBlob]) -> List[GaussianBlob]:
    """
    Sorts features in order of increasing scale-normalized Laplacian (meaning clearest feature first).

    :param blobs: A list of Gaussian blobs.
    :return: The same list of blobs, but sorted in order of increasing log.
    """
    blobs_increasing_log = copy.deepcopy(blobs)
    blobs_increasing_log.sort(key=lambda b: b.log)

    return blobs_increasing_log
