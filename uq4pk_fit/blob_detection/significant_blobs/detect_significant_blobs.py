
import numpy as np
from typing import List, Tuple, Union, Sequence

from uq4pk_fit.blob_detection.detect_blobs import compute_overlap, detect_blobs
from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob
from uq4pk_fit.blob_detection.blankets.second_order_blanket import second_order_blanket
from ..scale_normalized_laplacian import scale_normalized_laplacian


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
    :returns: A list of triplets. The first element of each triplet is a blob detected in the MAP estimate. It is an
        :py:class:`GaussianBlob` object. The second element is either None (if the blob is not significant), or another
        Gaussian blob, representing the significant feature. The third element is either None (if the blob is not
        significant), or the scale at which the significance was detected.
    """
    # Check input for consistency.
    assert lower_stack.ndim == 3 == upper_stack.ndim
    assert lower_stack.shape == upper_stack.shape
    s, m, n = lower_stack.shape
    assert reference.shape == (m, n)
    assert len(sigma_list) == s

    # Translate sigma_list to scale_list
    scale_list = [0.5 * np.sum(np.square(sigma)) for sigma in sigma_list]

    # Identify features in reference image.
    reference_blobs = detect_blobs(image=reference, sigma_list=sigma_list, max_overlap=overlap, rthresh=rthresh)
    # Compute blanket stack.
    blanket_stack = _compute_blanket_stack(lower_stack=lower_stack, upper_stack=upper_stack)
    # Apply scale-normalized Laplacian to blanket stack.
    blanket_laplacian_stack = scale_normalized_laplacian(ssr=blanket_stack, scales=scale_list, mode="reflect")
    # Compute mapped pairs.
    significance_triplets = _match_blobs(sigma_list=sigma_list, reference_blobs=reference_blobs,
                                         blanket_laplacian_stack=blanket_laplacian_stack, overlap=overlap,
                                         rthresh=rthresh)

    # Return the mapped pairs.
    return significance_triplets


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


def _match_blobs(sigma_list: SigmaList, reference_blobs: List[GaussianBlob], blanket_laplacian_stack: np.ndarray,
                 overlap: float, rthresh: float) -> List[Tuple]:
    """
    For a given blanket-feature array, determines whether any features "match" the features in ``map_features``
    at the given resolution.

    :param reference_blobs: The blobs in the reference image.
    :param blanket_stack: The stack of second-order blankets.
    :returns: The list of mapped pair. Each mapped pair is a tuple of the form (b, c) or (b, None). In the former case,
        b gives the MAP blob and c the associated significant blob. In the latter case, the MAP blob was not
        found to be significant.
    """
    n_blobs = len(reference_blobs)
    significant_blobs = [None for blob in reference_blobs]
    significant_scales = [None for blob in reference_blobs]

    # Iterate over log_stack
    for sigma, blanket in zip(sigma_list, blanket_stack):
        # Determine blobs in blanket.
        blanket_blobs = detect_blobs(image=blanket, sigma_list=sigma_list, max_overlap=overlap, rthresh=rthresh)
        #plot_blobs(image=blanket, blobs=blanket_blobs, show=True)
        n_significant = len(blanket_blobs)
        print(f"Found {n_significant} blanket-blobs.")
        # Match to blobs in MAP
        print(f"Scale: {sigma}.")
        for i in range(n_blobs):
            blob_i = reference_blobs[i]
            match_i = significant_blobs[i]
            new_match = _find_blob(blob_i, blanket_blobs, overlap=overlap)
            if match_i is None and new_match is not None:
                print(f"Significant blob found at {new_match.position}")
                significant_blobs[i] = new_match
                significant_scales[i] = sigma
            elif new_match is not None:
                # If there is a match, compare to existing match. Replace if the new blob is stronger.
                if new_match.log < match_i.log:
                    print(f"Blob gets replaced. New position: {new_match.position}")
                    significant_blobs[i] = new_match
                    significant_scales[i] = sigma
            else:
                # Else, nothing happens
                pass

    # Now, we have to remove intersections.
    _remove_intersection(significant_blobs, overlap)

    # Generate output tuples
    significance_triplets = []
    for blob, significant_blob, sigma in zip(reference_blobs, significant_blobs, significant_scales):
        significance_triplets.append(tuple([blob, significant_blob, sigma]))

    return significance_triplets


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


def _remove_intersection(blobs: List[Union[GaussianBlob, None]], max_overlap: float):
    """
    Given a list of blobs, removes overlap. The blob with the smallest scale "wins". If both blobs have the same
    scale, the one with the best log wins.

    :param blobs: Of shape (k, 5). Each row corresponds to a feature and is of the form (w, h, i, j, snl).
    :param max_overlap: The maximum allowed overlap between features.
    """
    n_blobs = len(blobs)

    for i in range(n_blobs):
        blob_i = blobs[i]
        if blob_i is not None:
            # If blob is None, go through all blobs and compute intersection.
            for j in range(n_blobs):
                blob_j = blobs[j]
                if blob_j is not None:
                    # If intersection is above overlap, remove smaller blob.
                    overlap = compute_overlap(blob_i, blob_j)
                    if overlap > max_overlap:
                        if blob_i.scale > blob_j.scale:
                            blobs[i] = None
                        elif blob_i.scale < blob_j.scale:
                            blobs[j] = None
                        # (If scales are equal, both blobs can stay.)
