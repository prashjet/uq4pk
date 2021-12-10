
import numpy as np
from skimage import morphology
from typing import List, Tuple, Union

from uq4pk_fit.special_operators import DiscreteLaplacian

from ..fci import fci
from ..filter import SquaredExponentialFilterFunction
from ..linear_model import LinearModel
from .detect_features import compute_overlap, detect_features, threshold_local_minima, remove_overlap
from .minimize_bumps import minimize_bumps
from .scale_normalized_laplacian import scale_normalized_laplacian


RTHRESH = 0.05  # Relative threshold for scale-space minima
OTHRESH = 0.5   # Relative overlap-threshold for feature mapping. Features are mapped if overlap is larger than this.


def significant_blobs(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, sigma_min: float = 1,
                      sigma_max: float = 20, num_sigma: int = 8, k: int = None, ratio: float = 1.) \
        -> List[Tuple]:
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
    :param num_sigma: Number of intermediate discretization steps steps. For example, if ``r_min = 1``, ``r_max = 10`` and
        ``n_steps = 2``, the resolutions 1, 2.5, 5, 7.5 and 10 are checked.
    :param k: Radius of kernel. E.g. k=3 corresponds to a 7x7 kernel. If not provided, defaults to max(m, n).
    :param scaling: Determines the shape of the blobs.
    :returns: A list of tuples is returned. Each tuple corresponds to a detected feature.
        The first element of the tuple is the blob detected in the MAP estimate. It is of the form (w, h, i, j),
        where ``w`` is the width of the feature, ``h`` is its height, while ``(i, j)`` are the indices of its center.
        The second element of the tuple is the significant feature associated to the MAP feature. It is either of the
        same form (w, h, i, j), or None if the feature was not found to be significant.
    """
    # Check input for consistency.
    _check_input_significant_features(alpha, m, n, model, x_map, sigma_min, sigma_max)

    # Identify features in MAP estimate
    map_im = np.reshape(x_map, (m, n))
    map_features = detect_features(map_im, sigma_min, sigma_max, num_sigma=num_sigma, rthresh=RTHRESH,
                                   overlap=OTHRESH, ratio=ratio)

    # Determine the minimal radius at which a feature has been detected.
    min_sigma = np.min(map_features[:, -1])

    # Discretize the interval [r_min, r_max] into equidistant points.
    sigmas = _discretize_sigma(min_sigma, sigma_max, num_sigma)
    # scale = sigma^2 / 2
    resolutions = [0.5 * s ** 2 for s in sigmas]

    # Create blanket stack
    if k is None:
        k = max(m, n)
    blanket_stack = _compute_blanket_stack(alpha=alpha, m=m, n=n, model=model, x_map=x_map, scales=resolutions, k=k,
                                           ratio=ratio)

    # Using blanket stack, compute mapped pairs.
    mapped_pairs = _compute_mapped_pairs(blanket_stack=blanket_stack, resolutions=resolutions,
                                         map_features=map_features, ratio=ratio)

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
    :return: The list of radii.
    """
    # Check the input for sensibleness.
    assert sigma_min <= sigma_max
    assert num_sigma >= 0
    step_size = (sigma_max - sigma_min) / (num_sigma + 1)
    radii = [sigma_min + n * step_size for n in range(num_sigma + 2)]
    return radii


def _compute_blanket_stack(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, scales: List[float],
                           k: int, ratio: float) \
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
        blanket = _compute_blanket(alpha, m, n, model, x_map, t, k, ratio)
        blanket_list.append(blanket)
        counter += 1
    # Create blanket stack
    blanket_stack = np.array(blanket_list)
    return blanket_stack


def _compute_blanket(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, scale: float, k: int,
                     ratio: float)\
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
    :returns: Of shape (m, n). The computed blanket
    """
    # Setup Gaussian filter function.
    h_vec = np.array([scale, scale * (ratio ** 2)])
    filter_function = SquaredExponentialFilterFunction(m=m, n=n, a=1, b=1, c=k, d=k,
                                                       h=h_vec, boundary="zero")
    # Compute filtered credible interval.
    ci = fci(alpha=alpha, model=model, x_map=x_map, ffunction=filter_function)
    # Compute minimally bumpy element using taut_string
    lower = np.reshape(ci[:, 0], (m, n))
    upper = np.reshape(ci[:, 1], (m, n))
    blanket = minimize_bumps(lb=lower, ub=upper)
    # Assert that blanket has the right format before returning it.
    assert blanket.shape == (m, n)
    return blanket


def _compute_mapped_pairs(blanket_stack: np.ndarray, resolutions: List[float], map_features: np.ndarray, ratio: float)\
        -> List[Tuple]:
    # Compute scale-normalized Laplacian of blanket stack
    delta_norm_m = scale_normalized_laplacian(blanket_stack, resolutions, mode="reflect")

    # Determine local minima of delta_norm_M
    local_minima = morphology.local_minima(image=delta_norm_m, indices=True, allow_borders=True)
    local_minima = np.array(local_minima).T

    # Bring blobs in right format
    blob_list = []
    for b in local_minima:
        t = resolutions[b[0]]
        w = np.sqrt(2 * t)
        h = ratio * w
        snl = delta_norm_m[b[0], b[1], b[2]]
        blob_list.append(np.array([w, h, b[1], b[2], snl]))
    blobs = np.array(blob_list)

    # Remove local minima that are below threshold
    blobs = threshold_local_minima(blobs, rthresh=RTHRESH)

    # Remove overlap.
    blobs = remove_overlap(blobs, othresh=OTHRESH, ratio=ratio)

    # Sort the remaining features
    blobs = _best_feature_first(blobs)

    # Remove snl-column
    blobs = blobs[:, :-1]

    # Perform matching between the local minima of delta_norm_m and the MAP features.
    mapped_pairs = _match_to_map(blobs, map_features, ratio)

    # Check output
    for pair in mapped_pairs:
        b, c = pair
        assert b.shape == (4, )
        assert c is None or c.shape == (4,)

    return mapped_pairs


def _best_feature_first(features: np.ndarray) -> np.ndarray:
    """
    Sorts features in order of increasing scale-normalized Laplacian (meaning clearest feature first).

    :param features: Of shape (k, 5), where each row is of the form (w, h, i, j, snl)
    :param snl: 3-dimensional array, corresponding to the scale-normalized Laplacian.
    :return: The sorted array. Of shape (k, 5).
    """
    # Get the sorting permutation for the SNL-stack
    increasing_snl = np.argsort(features[:, -1])

    # Sort features in increasing order
    sorted_features = features[increasing_snl]

    return sorted_features


def _match_to_map(significant_features: np.ndarray, map_features: np.ndarray, ratio: float) -> List[Tuple]:
    """
    For a given blanket-feature array, determines whether any features "match" the features in ``map_features``
    at the given resolution.

    :param significant_features: Of shape (m, 4). The significant features. Each row corresponds to a feature and is
        of the form (w, h, i, j), where w, h are width and height and (i, j) the position.
    :param map_features: Of shape (n, 4). The features detected in the MAP estimate. Same structure as
        ``significant_features``.
    :param ratio: The height/width ratio.
    :returns: The list of mapped pair. Each mapped pair is a tuple of the form (b, c) or (b, None). In the former case,
        b gives the MAP feature and c the associated significant feature. In the latter case, the MAP feature was not
        found to be significant.
    """
    mapped_pairs = []
    # Iterate over the MAP features
    for map_feature in map_features:
        # Find the feature matching the map_feature
        significant_feature = _find_feature(map_feature, significant_features, othresh=OTHRESH, ratio=ratio)
        # Add corresponding pair to "mapped_pairs".
        mapped_pairs.append(tuple([map_feature, significant_feature]))
    return mapped_pairs


def _find_feature(feature: np.ndarray, features: np.ndarray, othresh: float, ratio: float) -> Union[np.ndarray, None]:
    """
    Find a feature in a given collection of features.
    A feature is mapped if the overlap to another feature is more than a given threshold.
    If there is more than one matching feature, then the FIRST MATCH is selected.
    If the relative overlap is 1 for more than one feature in ``features``, the first matching feature is selected.

    :param feature: Of shape (4, ). The feature to be found.
    :param features: Of shape (n, 4). The array of features in which ``feature`` is searched.
    :return: The mapped feature. If no fitting feature is found, None is returned.
    """
    # Check that input has right format.
    assert feature.shape == (4, )
    assert features.shape[1] == 4
    # Iterate over features.
    found = None
    for candidate in features:
        overlap = compute_overlap(feature[[0, 2, 3]], candidate[[0, 2, 3]], ratio)
        if overlap >= othresh:
            found = candidate
            break
    # Check that output has right format
    assert found is None or found.shape == (4, )
    return found
