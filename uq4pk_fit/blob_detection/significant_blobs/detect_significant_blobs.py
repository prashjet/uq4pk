
import numpy as np
from typing import List, Tuple, Union, Sequence

from uq4pk_fit.blob_detection.detect_blobs import compute_overlap, detect_blobs
from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob
from uq4pk_fit.blob_detection.blankets.second_order_blanket import second_order_blanket
from uq4pk_fit.visualization.plot_distribution_function import plot_distribution_function

from ..scale_normalized_laplacian import scale_normalized_laplacian
from ..detect_blobs import best_blob_first, stack_to_blobs


SHOW_BLANKETS = False


# Make type for sigma list
SigmaList = Sequence[Union[float, np.ndarray]]


def detect_interesting_blobs(sigma_list: SigmaList, reference: np.ndarray, regularized_stack: np.ndarray,
                             rthresh: float = 0.05, overlap1: float = 0.5, overlap2: float = 0.5):
    # Check input for consistency.
    assert regularized_stack.ndim == 3
    s, m, n = regularized_stack.shape
    assert reference.shape == (m, n)
    assert len(sigma_list) == s

    # Translate sigma_list to scale_list
    scale_list = [0.5 * np.sum(np.square(sigma)) for sigma in sigma_list]

    # Identify features in reference image.
    reference_blobs = detect_blobs(image=reference, sigma_list=sigma_list, max_overlap=overlap1, rthresh=rthresh)
    # Apply scale-normalized Laplacian to regularized stack.
    regularized_log_stack = scale_normalized_laplacian(ssr=regularized_stack, scales=scale_list, mode="reflect")
    # Compute blanket-blobs.
    interesting_blobs = stack_to_blobs(scale_stack=regularized_log_stack, sigma_list=sigma_list, rthresh=rthresh,
                                       max_overlap=overlap1, exclude_max_scale=True)
    # Compute mapped pairs.
    mapped_pairs = _match_blobs(reference_blobs=reference_blobs, blanket_blobs=interesting_blobs, overlap=overlap2)

    # Return the mapped pairs.
    return mapped_pairs


def detect_significant_blobs(sigma_list: SigmaList, lower_stack: np.ndarray,
                             upper_stack: np.ndarray, reference: np.ndarray, rthresh1: float = 0.05,
                             rthresh2: float = 0.1, overlap1: float = 0.5, overlap2: float = 0.5,
                             exclude_max_scale: bool = False)\
        -> List:
    """
    Performs uncertainty-aware blob blob_detection with automatic scale selection.

    Performs feature matching to determine which features in the MAP estimate are significant with respect to the
    posterior distribution defined by the given model.
    - A feature not present in ``ansatz`` cannot be significant.
    - Also, the size and position of the possible significant features are determined by ``ansatz``.
    - Hence, we only have to determine which features are significant, and the resolution of the signficant features.

    :param sigma_list: The list of standard deviations used for the FCIs.
    :param lower_stack: The stack of the lower-bound-images of the FCIs.
    :param upper_stack: The stack of the upper-bound-images of the FCIs.
    :param reference: The image for which significant features need to be determined, e.g. the MAP estimate.
    :param rthresh1: The relative threshold for feature strength that a blob has to satisfy in order to count as
        detected.
    :param rthresh2: A "significant" blob must have strength equal to more than the factor `rthresh2` of the strength
        of the corresponding MAP blob.
    :param overlap1: The maximum allowed overlap for blobs in the same image.
    :param overlap2: The relative overlap that is used in the matching of the blobs in the reference image to the
        blanket-blobs.
    :returns:  A list of tuples. The first element of each tuple is a blob detected in the MAP estimate. It is an
        :py:class:`GaussianBlob` object. The other element is either None (if the blob is not significant), or another
        Gaussian blob, representing the uncertainty.
    """
    # Check input for consistency.
    assert lower_stack.ndim == 3 == upper_stack.ndim
    assert lower_stack.shape == upper_stack.shape
    s, m, n = lower_stack.shape
    assert reference.shape == (m, n)
    assert len(sigma_list) == s

    # Translate sigma_list to scale_list
    scale_list = [0.25 * np.sum(np.square(sigma)) for sigma in sigma_list]
    # Identify features in reference image.
    reference_blobs = detect_blobs(image=reference, sigma_list=sigma_list, max_overlap=overlap1, rthresh=rthresh1)
    # Get LoG-value of weakest blob.
    log_thresh = max([blob.log for blob in reference_blobs])
    # Compute blanket stack.
    blanket_stack = _compute_blanket_stack(lower_stack=lower_stack, upper_stack=upper_stack)
    # Apply scale-normalized Laplacian to blanket stack.
    blanket_laplacian_stack = scale_normalized_laplacian(ssr=blanket_stack, scales=scale_list, mode="reflect")
    # Compute blanket-blobs.
    athresh = rthresh2 * log_thresh
    blanket_blobs = stack_to_blobs(scale_stack=blanket_laplacian_stack, sigma_list=sigma_list, athresh=athresh,
                                   max_overlap=overlap1, exclude_max_scale=exclude_max_scale)
    # Compute mapped pairs.
    mapped_pairs = _match_blobs(reference_blobs=reference_blobs, blanket_blobs=blanket_blobs, overlap=overlap2)

    # Return the mapped pairs.
    return mapped_pairs


def _compute_blanket_stack(lower_stack: np.ndarray, upper_stack: np.ndarray) \
        -> np.ndarray:
    """
    Computes the scale-space stack of blankets M[t, i, j] = f_t[i, j], where each blanket f_t is the solution of
    the optimization problem

    f_t = argmin int || \\nabla \\Laplace f(x)||_2^2 dx     s. t. l_t <= f_t <= u_t.

    :param lower_stack: Of shape (s, m, dim). The stack of lower bounds of the FCIs.
    :param upper_stack: Of shape (s, m, dim). The stack of upper bounds of the FCIs.
    :return: The blanket stack, of shape (s, m, dim). The first axis corresponds to scale, the other two axis to the
        image domain.
    """
    # Check input for consistency.
    assert lower_stack.ndim == 3 == upper_stack.ndim
    assert lower_stack.shape == upper_stack.shape

    blanket_list = []
    for lower, upper in zip(lower_stack, upper_stack):
        # Compute blanket at scale t.
        blanket = _compute_blanket(lower, upper)
        if SHOW_BLANKETS:
            plot_distribution_function(image=blanket, show=True)
        blanket_list.append(blanket)
    # Return blanket stack as array.
    blanket_stack = np.array(blanket_list)
    # Check that blanket stack has correct format
    assert blanket_stack.shape == lower_stack.shape

    return blanket_stack


def _compute_blanket(lower: np.ndarray, upper: np.ndarray)\
        -> np.ndarray:
    """
    Compute blankets at given resolution for the given model.

    :param lower: Of shape (m, dim). The lower bound of the FCI.
    :param upper: Of shape (m, dim). The upper bound of the FCI.
    :returns: Of shape (m, dim). The computed blanket
    """
    # Check input.
    assert lower.ndim == 2 == upper.ndim
    assert lower.shape == upper.shape

    # Compute second order blanket argmin_f ||nabla delta f||_2^2 s.t. lower <= f <= upper.
    blanket = second_order_blanket(lb=lower, ub=upper)
    # Assert that blanket has the right format before returning it.
    assert blanket.shape == lower.shape

    return blanket


def _match_blobs(reference_blobs: List[GaussianBlob], blanket_blobs: List[GaussianBlob], overlap: float)\
        -> List[Tuple]:
    """
    Given a set of reference blobs and a set of blanket-blobs, we look for matches.

    :param reference_blobs: The blobs in the reference image.
    :param blanket_blobs: The blobs that can be identified in the stack of t-blankets.
    :param overlap: If the relative overlap between a reference blob and a blanket-blob is above this value, then this
        counts as a map.
    :returns: The list of mapped pair. Each mapped pair is a tuple of the form (b, c) or (b, None). In the former case,
        b gives the MAP blob and c the associated significant blob. In the latter case, the MAP blob was not
        found to be significant.
    """
    blanket_blobs_sorted = best_blob_first(blanket_blobs)

    mapped_pairs = []
    # Iterate over the MAP features
    for blob in reference_blobs:
        # Find the feature matching the map_feature
        matching_blob = _find_blob(blob, blanket_blobs_sorted, overlap=overlap)
        # Check that blobs really have the right overlap
        if matching_blob is not None:
            overl = compute_overlap(blob, matching_blob)
            assert overl >= overlap
        # Add corresponding pair to "mapped_pairs".
        mapped_pairs.append(tuple([blob, matching_blob]))

    return mapped_pairs


def _minima_to_blobs(minima: np.ndarray, log: np.array, thresh: float, sigma: Union[float, np.ndarray]):
    """
    Translates local minima of LoG image into blobs.

    :param minima:
    :param log:
    :return:
    """
    blobs = []
    for minimum in minima:
        log_value = log[minimum[0], minimum[1]]
        if log_value <= thresh:
            blob = GaussianBlob(x1=minimum[0], x2=minimum[1], sigma=sigma, log=log_value)
            blobs.append(blob)

    return blobs


def _find_blob(blob: GaussianBlob, blobs: List[GaussianBlob], overlap: float) \
        -> Union[GaussianBlob, None]:
    """
    Find a feature in a given collection of features.
    A feature is mapped if the overlap to another feature is more than a given threshold.
    If there is more than one matching feature, then the FIRST MATCH is selected.
    If the relative overlap is 1 for more than one feature in ``features``, the first matching feature is selected.
    """
    # Iterate over blobs.
    found = None
    for candidate in blobs:
        candidate_overlap = compute_overlap(blob, candidate)
        if candidate_overlap >= overlap:
            found = candidate
            break
    return found
