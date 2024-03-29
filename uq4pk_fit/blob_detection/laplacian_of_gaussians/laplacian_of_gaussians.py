
import copy
import numpy as np
from typing import List, Sequence, Union
from skimage import morphology

from uq4pk_fit.gaussian_blob import GaussianBlob, scale_space_representation, scale_normalized_laplacian
from uq4pk_fit.blob_detection.blob_geometry import compute_overlap


SigmaList = Sequence[Union[float, np.ndarray]]


def laplacian_of_gaussians(image: np.ndarray, sigma_list: SigmaList, max_overlap: float = 0.5, rthresh: float = 0.01,
                           exclude_max_scale: bool = False) -> List[GaussianBlob]:
    """
    Detects blobs in an image using the difference-of-Gaussians method.
    See https://en.wikipedia.org/wiki/Difference_of_Gaussians.

    Parameters
    ----------
    image :
        The image as 2d-array.
    sigma_list :
        List of the standard deviations used for the Gaussian blobs.
    max_overlap :
        If two blobs have a relative overlap larger than this number, they are considered as one.
    rthresh :
         The relative threshold for detection of blobs. A blob is only detected if it corresponds to a
        scale-space minimum of the scale-normalized Laplacian that is below ``rthresh * log_stack.min()``, where
        ``log_stack`` is the stack of Laplacian-of-Gaussians.
    exclude_max_scale :
        If True, the scale-space minima at the largest scale are ignored. This
        can sometimes improve the results of the blob detection.

    Returns
    -------
    result : list
        Returns a list of GaussianBlob-objects, each representing one detected blob.
    """
    # Check input for consistency.
    assert image.ndim == 2

    # COMPUTE LOG-STACK
    # Compute scale-space representation.
    ssr = scale_space_representation(image=image, sigmas=sigma_list)
    # Evaluate scale-normalized Laplacian
    log_stack = scale_normalized_laplacian(ssr, sigma_list)

    # Determine scale-space blobs as local scale-space minima
    blobs = stack_to_blobs(scale_stack=ssr, log_stack=log_stack, sigma_list=sigma_list, rthresh=rthresh,
                           max_overlap=max_overlap, exclude_max_scale=exclude_max_scale)

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

    Parameters
    ---
    blobs : list
    max_overlap : The maximum allowed relative overlap for blobs.

    Returns
    ---
    cleaned_blobs : list
        The blobs with overlap removed.
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


def stack_to_blobs(scale_stack: np.ndarray, log_stack: np.ndarray, sigma_list: SigmaList,
                   max_overlap: Union[float, None], exclude_max_scale: bool, athresh: float = None,
                   rthresh: float = None)\
        -> List[GaussianBlob]:
    """
    Given a scale-space stack, detects blobs as scale-space minima.
    """
    assert not (rthresh is None and athresh is None)

    # DETERMINE SCALE-SPACE BLOBS
    # Determine local scale-space minima
    local_minima = morphology.local_minima(image=log_stack, indices=True, allow_borders=True)
    local_minima = np.array(local_minima).T

    n_scales = scale_stack.shape[0]
    # If exclude_borders is True, we remove the local minima at the largest scale.
    if exclude_max_scale:
        local_minima = np.array([minimizer for minimizer in local_minima if minimizer[0] != n_scales - 1])

    if local_minima.size == 0:
        blobs = []
    else:
        # Bring output in correct format.
        blobs = []
        for b in local_minima:
            sigma_b = sigma_list[b[0]]
            log_b = log_stack[b[0], b[1], b[2]]
            blob = GaussianBlob(x1=b[1], x2=b[2], sigma=sigma_b, log=log_b)
            blobs.append(blob)

        # Remove all features below threshold.
        # If 'athresh' is given, it is used instead of 'rthresh'.
        if athresh is None:
            athresh = log_stack.min() * rthresh
        blobs = threshold_local_minima(blobs, athresh)

        # Remove overlap
        if max_overlap is not None:
            blobs = remove_overlap(blobs, max_overlap)

    return blobs


def best_blob_first(blobs: List[GaussianBlob]) -> List[GaussianBlob]:
    """
    Sorts blobs in order of increasing scale-normalized Laplacian (meaning clearest feature first).
    """
    blobs_increasing_log = copy.deepcopy(blobs)
    blobs_increasing_log.sort(key=lambda b: b.log)

    return blobs_increasing_log
