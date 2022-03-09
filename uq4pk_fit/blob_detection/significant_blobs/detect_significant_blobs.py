
import numpy as np
from typing import List, Tuple, Union, Sequence

from uq4pk_fit.blob_detection.detect_blobs import best_blob_first, compute_overlap, detect_blobs, stack_to_blobs
from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob
from uq4pk_fit.blob_detection.blankets.second_order_blanket import second_order_blanket
from uq4pk_fit.blob_detection.scale_normalized_laplacian import scale_normalized_laplacian


# Make type for sigma list
SigmaList = Sequence[Union[float, np.ndarray]]


def detect_significant_blobs(sigma_list: SigmaList, lower_stack: np.ndarray,
                             upper_stack: np.ndarray, reference: np.ndarray, rthresh: float = 0.05,
                             overlap: float = 0.5)\
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
    :param rthresh: The relative threshold for filtering out weak features.
    :param overlap: The maximum allowed overlap for detected features.
    :returns: A list of tuples. The first element of each tuple is a blob detected in the MAP estimate. It is an
        :py:class:`GaussianBlob` object. The other element is either None (if the blob is not significant), or another
        Gaussian blob, representing the significant feature.
    """
    # Check input for consistency.
    assert lower_stack.ndim == 3 == upper_stack.ndim
    assert lower_stack.shape == upper_stack.shape
    s, m, n = lower_stack.shape
    assert reference.shape == (m, n)
    assert len(sigma_list) == s

    # Identify features in reference image.
    map_blobs = detect_blobs(image=reference, sigma_list=sigma_list, max_overlap=overlap, rthresh=rthresh,
                             mode="constant")
    # Compute blanket stack.
    blanket_stack = _compute_blanket_stack(lower_stack=lower_stack, upper_stack=upper_stack)
    # Compute significant blobs
    laplacian_blanket_stack = scale_normalized_laplacian(blanket_stack, sigma_list, mode="reflect")
    # Compute mapped pairs.
    mapped_pairs = _match_blobs(sigma_list=sigma_list, map_blobs=map_blobs, log_stack=laplacian_blanket_stack,
                                overlap=overlap, rthresh=rthresh)

    # Return the mapped pairs.
    return mapped_pairs


def _discretize_sigma(sigma_min: float, sigma_max: float, num_sigma: int) -> List[float]:
    """
    Returns list of resolutions of a constant ratio, such that the range [min_scale, max_scale] is covered.

    :param sigma_min: Minimum resolution.
    :param sigma_max: Maximum resolution.
    :param num_sigma: Number of intermediate steps.
    :return: The list of sigma values.
    """
    # Check the input for sensibleness.
    assert sigma_min <= sigma_max
    assert num_sigma >= 0
    step_size = (sigma_max - sigma_min) / (num_sigma + 1)
    sigmas = [sigma_min + n * step_size for n in range(num_sigma + 2)]

    return sigmas


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


def _match_blobs(sigma_list: SigmaList, map_blobs: List[GaussianBlob], log_stack: np.ndarray, overlap: float,
                 rthresh: float) -> List[Tuple]:
    """
    For a given blanket-feature array, determines whether any features "match" the features in ``map_features``
    at the given resolution.

    :param map_blobs: The blobs in the reference image.
    :param log_stack: The Laplacian of Gaussian blob.
    :returns: The list of mapped pair. Each mapped pair is a tuple of the form (b, c) or (b, None). In the former case,
        b gives the MAP blob and c the associated significant blob. In the latter case, the MAP blob was not
        found to be significant.
    """
    mapped_pairs = []
    # Iterate over the MAP features
    for map_blob in map_blobs:
        # Remove all scales below the scale of ``map_blob``.
        log_stack_cut = log_stack[map_blob._scaleno:]
        sigma_list_cut = sigma_list[map_blob._scaleno:]
        # Detect significant features in log.
        significant_blobs = stack_to_blobs(sigma_list=sigma_list_cut, scale_stack=log_stack_cut, rthresh=rthresh,
                                           max_overlap=overlap)
        # Sort the significant features in order of increasing log.
        blobs_increasing_log = best_blob_first(significant_blobs)
        # Find the feature matching the map_feature
        significant_blob = _find_blob(map_blob, blobs_increasing_log, overlap=overlap)
        # Check that blobs really have the right overlap
        if significant_blob is not None:
            overl = compute_overlap(map_blob, significant_blob)
            print(f"Map blob = {map_blob.vector}, significant_blob = {significant_blob.vector}")
            print(f"Overlap = {overl}")
            assert overl >= overlap
        # Add corresponding pair to "mapped_pairs".
        mapped_pairs.append(tuple([map_blob, significant_blob]))

    return mapped_pairs


def _find_blob(blob: GaussianBlob, blobs: List[GaussianBlob], overlap: float) -> Union[GaussianBlob, None]:
    """
    Find a feature in a given collection of features.
    A feature is mapped if the overlap to another feature is more than a given threshold.
    If there is more than one matching feature, then the FIRST MATCH is selected.
    If the relative overlap is 1 for more than one feature in ``features``, the first matching feature is selected.

    :param blob: Of shape (4, ). The feature to be found.
    :param blobs: Of shape (dim, 4). The array of features in which ``feature`` is searched.
    :return: The mapped feature. If no fitting feature is found, None is returned.
    """
    # Iterate over blobs.
    found = None
    for candidate in blobs:
        candidate_overlap = compute_overlap(blob, candidate)
        if candidate_overlap >= overlap:
            found = candidate
            break
    return found
