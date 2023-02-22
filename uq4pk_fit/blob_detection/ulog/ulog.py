
import numpy as np
from typing import List, Union, Sequence
from uq4pk_fit.blob_detection.laplacian_of_gaussians.laplacian_of_gaussians import compute_overlap, \
    laplacian_of_gaussians
from uq4pk_fit.gaussian_blob import GaussianBlob, scale_normalized_laplacian
from uq4pk_fit.blob_detection.blankets.compute_blanket import compute_blanket
from uq4pk_fit.blob_detection.laplacian_of_gaussians.laplacian_of_gaussians import best_blob_first, stack_to_blobs


# Make type for sigma list
SigmaList = Sequence[Union[float, np.ndarray]]


def ulog(sigma_list: SigmaList, lower_stack: np.ndarray, upper_stack: np.ndarray, reference: np.ndarray,
         rthresh1: float = 0.05, rthresh2: float = 0.1, overlap1: float = 0.5, overlap2: float = 0.5,
         exclude_max_scale: bool = False) -> List:
    """
    Performs uncertainty-aware blob blob_detection with automatic scale selection.

    Performs feature matching to determine which features in the MAP estimate are significant with respect to the
    posterior distribution defined by the given model.
    - A feature not present in ``ansatz`` cannot be significant.
    - Also, the size and position of the possible significant features are determined by ``ansatz``.
    - Hence, we only have to determine which features are significant, and the resolution of the signficant features.

    Parameters
    ----------
    sigma_list :
        The list of standard deviations.
    lower_stack :
        The stack of the lower-bound-images of the FCIs.
    upper_stack :
        The stack of the upper-bound-images of the FCIs.
    reference :
        The image for which significant features need to be determined, e.g. the MAP estimate.
    rthresh1:
        The relative threshold for feature strength that a blob has to satisfy in order to count as detected.
    rthresh2 :
        A "significant" blob must have strength equal to more than the factor `rthresh2` of the strength
        of the corresponding MAP blob.
    overlap1 :
        The maximum allowed overlap for blobs in the same image.
    overlap2 : The relative overlap that is used in the matching of the blobs in the reference image to the
        blanket-blobs.
    exclude_max_scale : bool
        If True, blobs cannot be detected at the maximal scale.

    Returns
    -------
    mapped_pairs : list
        A list of tuples. The first element of each tuple is a blob detected in the MAP estimate. It is an
        `GaussianBlob` object. The other element is either None (if the blob is not significant), or another
        Gaussian blob, representing the uncertainty.
    """
    # Check input for consistency.
    assert lower_stack.ndim == 3 == upper_stack.ndim
    assert lower_stack.shape == upper_stack.shape
    s, m, n = lower_stack.shape
    assert reference.shape == (m, n)
    assert len(sigma_list) == s

    # Identify features in reference image.
    reference_blobs = laplacian_of_gaussians(image=reference, sigma_list=sigma_list, max_overlap=overlap1,
                                             rthresh=rthresh1)
    # Compute blanket stack.
    blanket_stack = _compute_blanket_stack(lower_stack=lower_stack, upper_stack=upper_stack)
    # Apply scale-normalized Laplacian to blanket stack.
    blanket_laplacian_stack = scale_normalized_laplacian(ssr=blanket_stack, sigmas=sigma_list)
    # Compute blanket-blobs.
    blanket_blobs = stack_to_blobs(scale_stack=blanket_stack, log_stack=blanket_laplacian_stack, sigma_list=sigma_list,
                                   rthresh=rthresh2, max_overlap=overlap1, exclude_max_scale=exclude_max_scale)
    n_sig = len(blanket_blobs)
    print(f"{n_sig} significant blobs detected.")
    # Compute mapped pairs.
    mapped_pairs, n_mapped = _match_blobs(reference_blobs=reference_blobs, blanket_blobs=blanket_blobs,
                                          overlap=overlap2, rthresh=rthresh2)
    n_ref = len(reference_blobs)
    print(f"{n_mapped} matching reference blobs were found, out of {n_ref}.")

    # Return the mapped pairs.
    return mapped_pairs


def _compute_blanket_stack(lower_stack: np.ndarray, upper_stack: np.ndarray) -> np.ndarray:
    """
    Computes the scale-space stack of blankets M[t, i, j] = f_t[i, j], where each blanket f_t is the solution of
    the optimization problem

    f_t = argmin int || \\nabla \\Laplace f(x)||_2^2 dx     s. t. l_t <= f_t <= u_t.
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


def _compute_blanket(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Compute blankets at given scale for the given model.
    """
    # Check input.
    assert lower.ndim == 2 == upper.ndim
    assert lower.shape == upper.shape

    # Compute blanket argmin_f ||Lap(f)||_2^2 s.t. lower <= f <= upper.
    blanket = compute_blanket(lb=lower, ub=upper)
    # Assert that blanket has the right format before returning it.
    assert blanket.shape == lower.shape

    return blanket


def _match_blobs(reference_blobs: List[GaussianBlob], blanket_blobs: List[GaussianBlob], overlap: float,
                 rthresh: float):
    """
    Given a set of reference blobs and a set of blanket-blobs, we look for matches.
    """
    blanket_blobs_sorted = best_blob_first(blanket_blobs)

    mapped_pairs = []
    # Iterate over the MAP features
    n_mapped = 0
    for blob in reference_blobs:
        # Find the feature matching the map_feature
        matching_blob = _find_blob(blob, blanket_blobs_sorted, overlap=overlap, rthresh=rthresh)
        # Check that blobs really have the right overlap
        if matching_blob is not None:
            overl = compute_overlap(blob, matching_blob)
            n_mapped += 1
            assert overl >= overlap
        # Add corresponding pair to "mapped_pairs".
        mapped_pairs.append(tuple([blob, matching_blob]))

    return mapped_pairs, n_mapped


def _minima_to_blobs(minima: np.ndarray, log: np.array, thresh: float, sigma: Union[float, np.ndarray]):
    """
    Translates local minima of LoG image into blobs.
    """
    blobs = []
    for minimum in minima:
        log_value = log[minimum[0], minimum[1]]
        if log_value <= thresh:
            blob = GaussianBlob(x1=minimum[0], x2=minimum[1], sigma=sigma, log=log_value)
            blobs.append(blob)

    return blobs


def _find_blob(blob: GaussianBlob, blobs: List[GaussianBlob], overlap: float, rthresh: float) \
        -> Union[GaussianBlob, None]:
    """
    Find a feature in a given collection of features.
    A feature is mapped if the overlap to another feature is more than a given threshold.
    If there is more than one matching feature, then the FIRST MATCH is selected.
    If the relative overlap is 1 for more than one feature in ``features``, the first matching feature is selected.
    """
    found = None
    for candidate in blobs:
        candidate_overlap = compute_overlap(blob, candidate)
        enough_overlap = (candidate_overlap >= overlap)
        enough_log = (candidate.log <= rthresh * blob.log)
        if enough_overlap and enough_log:
            found = candidate
            break
    return found
