
import numpy as np
from typing import List, Literal, Tuple, Union

from uq4pk_fit.uq_mode.fci import fci
from uq4pk_fit.uq_mode.discretization import LocalizationWindows
from uq4pk_fit.uq_mode.filter import GaussianFilterFunction2D
from uq4pk_fit.uq_mode.linear_model import LinearModel
from uq4pk_fit.blob_detection.detect_blobs import best_blob_first, compute_overlap, detect_blobs, stack_to_blobs
from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob
from uq4pk_fit.blob_detection.blankets.second_order_blanket import second_order_blanket
from uq4pk_fit.blob_detection.scale_normalized_laplacian import scale_normalized_laplacian


def detect_significant_blobs(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, sigma_min: float = 1,
                             sigma_max: float = 20, num_sigma: int = 8, k: int = None, ratio: float = 1.,
                             rthresh: float = 0.05, overlap: float = 0.5,
                             blanket_mode: Literal["fast", "exact"] = "fast") -> List:
    """
    Performs uncertainty-aware blob blob_detection with automatic scale selection.

    Performs feature matching to determine which features in the MAP estimate are significant with respect to the
    posterior distribution defined by the given model.
    - A feature not present in ``ansatz`` cannot be significant.
    - Also, the size and position of the possible significant features are determined by ``ansatz``.
    - Hence, we only have to determine which features are significant, and the resolution of the signficant features.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The statistical mode. Must be a :py:class:`LinearModel` object.
    :param x_map: The MAP estimate.
    :param sigma_min: The minimal radius at which features can be detected.
    :param sigma_max: The maximal radius at which features can be detected.
    :param num_sigma: Number of intermediate discretization steps steps. For example, if ``sigma_min = 1``,
        ``sigma_max = 10`` and ``num_sigma = 3``, the resolutions 1, 2.5, 5, 7.5 and 10 are checked.
    :param k: The truncation radius of the Gaussian filter kernel. E.g. k=3 corresponds to a 7x7 kernel.
        If not provided, defaults to max(m, n).
    :param ratio: The height / width ratio for each blob.
    :param rthresh: The relative threshold for filtering out weak features.
    :param overlap: The maximum allowed overlap for detected features.
    :param blanket_mode: Mode for computing the second-order blankets.
    :returns: A list of tuples. The first element of each tuple is a blob detected in the MAP estimate. It is an
        :py:class:`GaussianBlob` object. The other element is either None (if the blob is not significant), or another
        Gaussian blob, representing the significant feature.
    """
    # Check input for consistency.
    _check_input_significant_features(alpha, m, n, model, x_map, sigma_min, sigma_max)

    # Identify features in MAP estimate
    map_im = np.reshape(x_map, (m, n))
    map_blobs = detect_blobs(map_im, sigma_min, sigma_max, num_sigma=num_sigma, max_overlap=overlap, rthresh=rthresh,
                             mode="constant", ratio=ratio)

    # Determine the minimal sigma at which a feature has been detected.
    sigma_list = [blob._sigma_x for blob in map_blobs]
    min_sigma = np.min(np.array(sigma_list))

    # Discretize the interval [r_min, r_max] into equidistant points.
    sigmas = _discretize_sigma(min_sigma, sigma_max, num_sigma)
    # scale = sigma^2 / 2
    resolutions = [0.5 * s ** 2 for s in sigmas]

    # Create blanket stack
    if k is None:
        k = max(m, n)
    blanket_stack = _compute_blanket_stack(alpha=alpha, m=m, n=n, model=model, x_map=x_map, scales=resolutions, k=k,
                                           ratio=ratio, blanket_mode=blanket_mode)

    # Using blanket stack, compute mapped pairs.
    mapped_pairs = _compute_mapped_pairs(blanket_stack=blanket_stack, resolutions=resolutions, map_blobs=map_blobs,
                                         ratio=ratio, rthresh=rthresh, max_overlap=overlap)

    # Return the mapped pairs.
    return mapped_pairs


def _check_input_significant_features(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray,
                                      minscale: float, maxscale: float):
    if not 0 < alpha < 1:
        raise Exception(f"Must have 0 < alpha < 1 ('alpha' is {alpha}).")
    if not m * n == x_map.size:
        raise Exception(f"x_map.size must equal m * n (x_map has size {x_map.size}, but m * n = {m * n}")
    if not model.n == x_map.size:
        raise Exception(f"'model' must have dimension {x_map.size}")
    if not minscale <= maxscale:
        raise Exception(f"minscale must not be larger than maxscale (minscale is {minscale} but maxscale is "
                        f"{maxscale}).")


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


def _compute_blanket_stack(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, scales: List[float],
                           k: int, ratio: float, blanket_mode: Literal["fast", "exact"]) \
        -> np.ndarray:
    """
    Computes the scale-space stack of blankets M[t, i, j] = f_t[i, j], where each blanket f_t is the solution of
    the optimization problem

    f_t = argmin int || \\nabla \\Laplace f(x)||_2^2 dx     s. t. l_t <= f_t <= u_t.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The statistical model (flattened).
    :param x_map: The MAP estimate (flattened).
    :param scales: The scales for which the blanket must be computed.
    :param k: Cutoff radius for convolution kernel. E.g. if k=3, we work with a 7x7 kernel.
    :param ratio: Height/width ratio.
    :return: The blanket stack. The first axis corresponds to scale, the other two axis to the image domain.
    """
    blanket_list = []
    counter = 1
    for t in scales:
        print(f"Scale {counter} / {len(scales)}.")
        # Compute blanket at scale t.
        blanket = _compute_blanket(alpha, m, n, model, x_map, t, k, ratio, blanket_mode)
        blanket_list.append(blanket)
        counter += 1
    # Create blanket stack
    blanket_stack = np.array(blanket_list)

    return blanket_stack


def _compute_blanket(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, scale: float, k: int,
                     ratio: float, blanket_mode: Literal["fast", "exact"])\
        -> np.ndarray:
    """
    Compute blankets at given resolution for the given model.

    :param alpha:
    :param m:
    :param n:
    :param model:
    :param x_map:
    :param scale: The scale at which the blanket should be computed.
    :param k: The kernel cutoff radius.
    :param ratio:
    :param blanket_mode:
    :returns: Of shape (m, n). The computed blanket
    """
    # Setup Gaussian filter function.
    sigma1 = 2 * np.sqrt(scale)
    sigma2 = sigma1 * ratio
    filter_function = GaussianFilterFunction2D(m=m, n=n, sigma1=sigma1, sigma2=sigma2, boundary="zero")
    discretization = LocalizationWindows(im_ref=x_map.reshape((m, n)), w1=k, w2=k)
    # Compute filtered credible interval.
    ci = fci(alpha=alpha, model=model, x_map=x_map, ffunction=filter_function, discretization=discretization)
    # Compute minimally bumpy element using taut_string
    lower = np.reshape(ci[:, 0], (m, n))
    upper = np.reshape(ci[:, 1], (m, n))
    blanket = second_order_blanket(lb=lower, ub=upper, mode=blanket_mode)

    # Assert that blanket has the right format before returning it.
    assert blanket.shape == (m, n)

    return blanket


def _compute_mapped_pairs(blanket_stack: np.ndarray, resolutions: List[float], map_blobs: List[GaussianBlob],
                          ratio: float, rthresh: float, max_overlap: float) -> List[Tuple]:
    # Compute scale-normalized Laplacian of blanket stack
    laplacian_blanket_stack = scale_normalized_laplacian(blanket_stack, resolutions, mode="reflect")

    sigmas = [np.sqrt(2 * t) for t in resolutions]
    sig_blobs = stack_to_blobs(scale_stack=laplacian_blanket_stack, sigmas=sigmas, rthresh=rthresh,
                               max_overlap=max_overlap, ratio=ratio)

    # Match the significant blobs to the MAP blobs.
    mapped_pairs = _match_to_map(sig_blobs, map_blobs, overlap=max_overlap)

    return mapped_pairs


def _match_to_map(significant_blobs: List[GaussianBlob], map_blobs: List[GaussianBlob], overlap: float) -> List[Tuple]:
    """
    For a given blanket-feature array, determines whether any features "match" the features in ``map_features``
    at the given resolution.

    :param significant_blobs:
    :param map_blobs:
    :returns: The list of mapped pair. Each mapped pair is a tuple of the form (b, c) or (b, None). In the former case,
        b gives the MAP blob and c the associated significant blob. In the latter case, the MAP blob was not
        found to be significant.
    """
    # Sort the significant features in order of increasing log.
    blobs_increasing_log = best_blob_first(significant_blobs)

    mapped_pairs = []
    # Iterate over the MAP features
    for map_blob in map_blobs:
        # Find the feature matching the map_feature
        significant_blob = _find_blob(map_blob, blobs_increasing_log, overlap=overlap)
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
    :param blobs: Of shape (n, 4). The array of features in which ``feature`` is searched.
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
