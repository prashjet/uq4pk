
import numpy as np
from skimage import morphology
from typing import List, Tuple, Union, Sequence

from uq4pk_fit.blob_detection.detect_blobs import best_blob_first, compute_overlap, detect_blobs
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
    reference_blobs = detect_blobs(image=reference, sigma_list=sigma_list, max_overlap=overlap, rthresh=rthresh,
                                   mode="constant")
    # Compute blanket stack.
    blanket_stack = _compute_blanket_stack(lower_stack=lower_stack, upper_stack=upper_stack)
    # Compute significant blobs
    laplacian_blanket_stack = scale_normalized_laplacian(blanket_stack, sigma_list, mode="reflect")
    # Compute mapped pairs.
    mapped_pairs = _match_blobs(sigma_list=sigma_list, reference_blobs=reference_blobs,
                                log_stack=laplacian_blanket_stack, overlap=overlap, rthresh=rthresh)

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


def _match_blobs(sigma_list: SigmaList, reference_blobs: List[GaussianBlob], log_stack: np.ndarray, overlap: float,
                 rthresh: float) -> List[Tuple]:
    """
    For a given blanket-feature array, determines whether any features "match" the features in ``map_features``
    at the given resolution.

    :param reference_blobs: The blobs in the reference image.
    :param log_stack: The Laplacian of Gaussian blob.
    :returns: The list of mapped pair. Each mapped pair is a tuple of the form (b, c) or (b, None). In the former case,
        b gives the MAP blob and c the associated significant blob. In the latter case, the MAP blob was not
        found to be significant.
    """
    matched_blobs = []
    unmatched_blobs = reference_blobs
    significant_blobs = []
    athresh = rthresh * log_stack.min()

    # Iterate over log_stack
    for sigma, log_image in zip(sigma_list, log_stack):
        # Determine blobs of LoG slice.
        local_minima = morphology.local_minima(image=log_image, indices=True)
        local_minima = np.array(local_minima).T
        # Turn minima to blobs
        significant_blobs_sigma = _minima_to_blobs(local_minima, log_image, thresh=athresh, sigma=sigma)
        # Remove all significant blobs that match matched blobs.
        significant_blobs_sigma_unmatched = []
        for significant_blob_sigma in significant_blobs_sigma:
            match_already = _find_blob(significant_blob_sigma, matched_blobs, overlap=overlap)
            if match_already is None:
                significant_blobs_sigma_unmatched.append(significant_blob_sigma)
        # Sort significant blobs in order of increasing LoG
        significant_blobs_sigma_unmatched = best_blob_first(significant_blobs_sigma_unmatched)

        # Match unmatched reference blobs with unmatched significant blobs.
        still_unmatched = []
        for i in range(len(unmatched_blobs)):
            blob_i = unmatched_blobs[i]
            significant_blob = _find_blob(blob_i, significant_blobs_sigma_unmatched, overlap=overlap)
            if significant_blob is not None:
                matched_blobs.append(blob_i)
                significant_blobs.append(significant_blob)
            else:
                still_unmatched.append(blob_i)
        # Update list of unmatched blobs
        unmatched_blobs = still_unmatched

    # Generate output tuples
    mapped_pairs = []
    for blob, significant_blob in zip(matched_blobs, significant_blobs):
        mapped_pairs.append(tuple([blob, significant_blob]))
    for blob in unmatched_blobs:
        mapped_pairs.append(tuple([blob, None]))

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
