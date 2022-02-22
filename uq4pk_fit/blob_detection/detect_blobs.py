
import copy
import numpy as np
from typing import List, Sequence, Union
from skimage import morphology

from .gaussian_blob import GaussianBlob
from .scale_space_representation import scale_space_representation
from .scale_normalized_laplacian import scale_normalized_laplacian
from .blob_geometry import compute_overlap


SigmaList = Sequence[Union[float, np.ndarray]]


def detect_blobs(image: np.ndarray, sigma_list: SigmaList, max_overlap: float = 0.5,
                 rthresh: float = 0.01, mode: str = "constant") -> List[GaussianBlob]:
    """
    Detects blobs in an image using the difference-of-Gaussians method.
    See https://en.wikipedia.org/wiki/Difference_of_Gaussians.

    :param image: The image as 2d-array.
    :param sigma_list: List of the standard deviations used for the Gaussian blobs. Each element must either be float
        or a (2,)-array, where the two numbers correspond to the standard deviation in vertical and horizontal
        direction.
    :param rthresh: The relative threshold for detection of blobs. A blob is only detected if it corresponds to a
        scale-space minimum of the scale-normalized Laplacian that is below ``rthresh * log_stack.min()``, where
        ``log_stack`` is the stack of Laplacian-of-Gaussians.
    :param max_overlap: If two blobs have a relative overlap larger than this number, they are considered as one.
    :param mode: Determines how the image boundaries are handled.
    :return: Returns a list of GaussianBlob-objects, each representing one detected blob.
    """
    # Check input for consistency.
    assert image.ndim == 2

    # COMPUTE LOG-STACK
    t_list = [0.5 * sigma ** 2 for sigma in sigma_list]
    # Compute scale-space representation.
    ssr = scale_space_representation(image=image, scales=t_list, mode=mode)
    # Evaluate scale-normalized Laplacian
    log_stack = scale_normalized_laplacian(ssr, t_list, mode="reflect")

    # Determine scale-space blobs as local scale-space minima
    blobs = stack_to_blobs(scale_stack=log_stack, sigma_list=sigma_list, rthresh=rthresh, max_overlap=max_overlap)

    return blobs


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


def stack_to_blobs(scale_stack: np.ndarray, sigma_list: SigmaList, rthresh: float, max_overlap: float)\
        -> List[GaussianBlob]:
    """
    Given a scale-space stack, detects blobs as scale-space minima.

    :param scale_stack:
    :param sigma_list: The list of standard deviations for each filter.
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
            sigma_b = sigma_list[b[0]]
            sslaplacian = scale_stack[b[0], b[1], b[2]]
            blob = GaussianBlob(x1=b[1], x2=b[2], sigma=sigma_b, log=sslaplacian)
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
