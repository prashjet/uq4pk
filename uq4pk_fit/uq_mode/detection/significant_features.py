
from math import acos, log, pi, sqrt
import numpy as np
from typing import List, Union

from ..fci import fci
from ..filter import SquaredExponentialFilterFunction
from ..linear_model import LinearModel
from .feature_detection import blob_dog
from .span_blanket import span_blanket

from .blob_plot import plot_blob


K = 1.6         # the scale ratio. See https://en.wikipedia.org/wiki/Difference_of_Gaussians.
RTOL = 0.05     # Relative tolerance for matching condition.
OTHRESH = 0.5   # Relative overlap-threshold for feature mapping. Features are mapped if overlap is larger than this.


def significant_features(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, minscale: float,
                         maxscale: float) -> np.ndarray:
    """
    Performs uncertainty-aware blob detection with automatic scale selection.

    Performs feature matching to determine which features in ``ansatz`` are significant with respect to
    ``blanket_list``, and at which resolution. For this, we note the following:
    - A feature not present in ``ansatz`` cannot be significant.
    - Also, the size and position of the possible significant features are determined by ``ansatz``.
    - Hence, we only have to determine which features are significant, and the resolution of the signficant features.

    This is implemented by iterating over all blankets, starting at the highest resolution (note that high resolution
    corresponds to low 'r').
    In the first step, we map the features in the current blanket to the features in ``ansatz``.
    Then, all features that do not match any feature in ``ansatz`` are dropped.
    Next, for each mapped feature we check a matching condition. The matching condition is
                                    s_a <= s <= max(s_a, r),
    where `s` is the size at which the feature is detected in the current blanket, `s_a` is the size of the
    corresponding feature in ``ansatz``, and `r` is the resolution of the current blanket.
    This matching condition is determined from the following rules.
    -   Since ``ansatz`` dictates the scale of all identifyable features, the detected scale of the feature in
        ``features``must be equal to the scale of the corresponding feature in ``ansatz``, with an exception when the
        resolution is larger than the scale of the feature in ``ansatz`` and the scale of the feature in ``feature``
        is less than the resolution. For example, this might happen when there is a very small feature in ``ansatz``
        that is only identifiable at a resolution larger than its size.
    If the matching condition is satisfied, then the feature is "significant" at the current resolution, and it is
    removed from the stack of features-to-be-identified.
    Afterwards, we proceed to the next mapped feature in the blanket. After we have checked all mapped features in the
    blanket, we check if all features in ``ansatz`` are significant. If yes, we are done. If not, we proceed with the
    blanket at the next-higher resolution.
    This iteration is continued until all features in ``ansatz`` are determined to be significant, or we arrive at
    the maximum scale.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The statistical mode. Must be a :py:class:`LinearModel` object.
    :param x_map: The MAP estimate.
    :param minscale: The minimal scale at which features should be detected.
    :param maxscale: The maximal scale at which features should be detected.
    :return: The detected features are returned as an array of shape (k, 4), where each row corresponds to a feature
        and is of the form (i, j, s, r), where (i, j) is the index in ``image`` at which the feature was identified,
        s is the detected scale of the feature, and ``r`` is the resolution at which the feature is detectable.
         If no features are detected, then None is returned.
    """
    # Check input for consistency.
    _check_input_significant_features(alpha, m, n, model, x_map, minscale, maxscale)
    # Identify features in MAP estimate
    map_im = np.reshape(x_map, (m, n))
    map_features = blob_dog(map_im, minscale, maxscale)
    # Select scales between minscale and maxscale, with a scale ratio of K.
    #  nscales is the largest n such that minscale * K ** n <= maxscale.
    nscales = np.floor((log(maxscale) - log(minscale)) / log(K)).astype(int)
    scales = [minscale * K ** i for i in range(nscales + 1)]
    # For each scale, compute the blanket and store in list
    blanket_list = _compute_blankets(alpha, m, n, model, x_map, scales)
    # On each blanket, run feature detection.
    features_list = []
    for blanket in blanket_list:
        features_of_blanket = blob_dog(blanket, minscale, maxscale)
        features_list.append(features_of_blanket)
    # Remove entries with no features.
    for i in range(len(scales)):
        if features_list[i] is None:
            features_list.pop(i)
            scales.pop(i)
    # Identify significant features by matching map_features with features_of_blankets
    features = _determine_resolution(map_features, features_list, scales)
    return features


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


def _compute_blankets(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, scales: List[float])\
        -> List[np.ndarray]:
    """
    Compute blankets at different scales for the given model.

    :param alpha:
    :param m:
    :param n:
    :param model:
    :param x_map:
    :param scales: The scales at which the blankets should be computed.
    :returns: A list of blankets as 2-dimensional numpy arrays.
    """
    blanket_list = []
    for scale in scales:
        print(f"At scale {scale}")
        # Setup Gaussian filter function.
        scale_int = np.ceil(scale).astype(int)
        filter_function = SquaredExponentialFilterFunction(m=m, n=n, a=1, b=1, c=2 * scale_int, d=2 * scale_int,
                                                           h=scale)
        ci = fci(alpha=alpha, model=model, x_map=x_map, ffunction=filter_function)
        # Compute minimally bumpy element using taut_string
        lower = np.reshape(ci[:, 0], (m, n))
        upper = np.reshape(ci[:, 1], (m, n))
        blanket = span_blanket(lb=lower, ub=upper)
        blanket_list.append(blanket)
    assert len(blanket_list) == len(scales)
    return blanket_list


def _determine_resolution(ansatz_features: np.ndarray, list_of_blanket_features: List[np.ndarray],
                          resolution_list: List[float]) -> Union[np.ndarray, None]:
    """

    :param ansatz_features: Of shape (k, 3).
    :param list_of_blanket_features: List with numpy arrays each of shape (k_j, 3), corresponding to the k_j features
        of the j-th blanket.
    :param resolution_list: List of same length as ``list_of_blanket_features``, giving for each blanket the corresponding
            resolutions.

    :return: The significant features are returned as array of shape (m, 4), where each row corresponds to a
        significant feature and is of the form (i, j, s, r), where (i, j) is the index of the feature, s is the scale,
        and r is the resolution. ``None`` is returned if no significant features are found.
    """
    # Check input consistency.
    assert ansatz_features.shape[1] == 3
    # Obtain local copy from ansatz_features so that we can change it.
    ansatz_features_loc = ansatz_features.copy()
    for blanket_features in list_of_blanket_features:
        assert blanket_features.shape[1] == 3
    assert len(list_of_blanket_features) == len(resolution_list)
    # Convert to arrays for convenience.
    arr_of_blanket_features = np.array(list_of_blanket_features)    # This is now a 3d-array.
    resolutions = np.array(resolution_list)
    # First, we sort blankets in order of decreasing resolution (increasing r).
    decreasing_resolution = np.argsort(resolutions)
    resolutions_sorted = resolutions[decreasing_resolution]
    arr_of_blanket_features = arr_of_blanket_features[decreasing_resolution]
    # Initialize output list
    output_list = []
    # Then, we iterate over blankets.
    for blanket_features, resolution in zip(arr_of_blanket_features, resolutions_sorted):
        # First, map blanket features to ansatz features, obtaining a list of feature pairs.
        mapped_pairs = _map_features(blanket_features, ansatz_features_loc)
        # For each feature-pair, check the matching condition.
        for blanket_feature, ansatz_feature in mapped_pairs:
            features_match = _matching_condition(blanket_feature, ansatz_feature, resolution)
            # If the matching condition is satisfied, remove the corresponding feature from ``ansatz``...
            if features_match:
                ansatz_features_loc = _remove_feature(ansatz_features_loc, ansatz_feature)
                # ... and add the significant feature to the output list, with the current resolution as 4-th entry.
                significant = np.append(ansatz_feature, resolution)
                output_list.append(significant)
        # If there are no features left in ``ansatz``, we are done.
        if ansatz_features.size == 0:
            break
    # Form an array of the right format from the output list.
    if len(output_list) == 0:
        output = None
    else:
        output = np.array(output_list)
        # Perform a sanity check on the output array.
        m = output.shape[0]
        assert output.shape == (m, 4)
    # Return that array.
    return output


def _map_features(features_of_blanket: np.ndarray, ansatz_features: np.ndarray) -> List:
    """
    Maps blanket features to Ansatz features.
    A feature is mapped if the overlap to another feature is more than a given threshold.

    :param features_of_blanket: Of shape (k, 3).
    :param ansatz_features: Of shape (n, 3)
    :return: The mapped features as list of tuples of the form (blanket_feature, ansatz_feature), where both
        blanket_feature and ansatz_feature have the form (i, j, s), with (i, j) the index and s the scale.
        Returns an empty list if no features have been mapped.
    """
    # Check that input has right format.
    assert features_of_blanket.shape[1] == 3
    assert ansatz_features.shape[1] == 3
    # Copy ansatz_features since we will remove entries.
    ansatz_features_cp = ansatz_features.copy()
    # Initialize output list.
    output_list = []
    # Iterate over features in blanket.
    for blanket_feature in features_of_blanket:
        # Compute overlap with each feature in ansatz. If overlap is larger than threshold, the feature is mapped.
        for i in range(ansatz_features_cp.shape[0]):
            ansatz_feature = ansatz_features_cp[i]
            overlap = _compute_overlap(blanket_feature, ansatz_feature)
            if overlap >= OTHRESH:
                # Remove feature from ansatz_features ...
                ansatz_features_cp = np.delete(ansatz_features_cp, i, axis=0)
                # ... and add mapped pair to output list.
                pair = (blanket_feature, ansatz_feature)
                output_list.append(pair)
                break
    # Return output list.
    return output_list


def _compute_overlap(feature1: np.ndarray, feature2: np.ndarray):
    """
    Computes relative overlap of two features (i, j, r) and (k, l, s).
    The relative overlap is I / a,
    where a is the area of the smaller circle, and I is the size of the intersection, i.e.
    I = 0, if d > s + r,
    I = 1, if d <= max(r-s, s-r),
    I = r1^2 acos(d_1 / r2) - d_1 sqrt(r1^2 - d_1^2) + r^2 acos(d_2 / r2) - d_w sqrt(r1^2 - d_2^2), otherwise,
    where d_1 = (r1^2 - r2^2 + d^2) / (2 d), d_2 = d - d_1, d is the distance of the features, and
    See https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6 for a derivation.

    :param feature1: Of shape (3, ).
    :param feature2: Of shape (3, ).
    :return: The relative overlap, a number between 0 and 1.
    """
    # Check input for right format.
    assert feature1.shape == feature2.shape == (3, )
    # Compute the distance of the two circles.
    pos1 = feature1[:2]
    pos2 = feature2[:2]
    # r1 >= r2
    s = feature1[-1]
    r = feature2[-1]
    assert s > 0 and r > 0
    r1 = max(s, r)
    r2 = min(s, r)
    d = np.linalg.norm(pos1 - pos2)
    # Compute relative overlap
    if d > s + r:
        relative_overlap = 0.
    elif d <= r1 - r2:
        relative_overlap = 1
    else:
        # Compute area of smaller circle.
        a = pi * r2 ** 2
        # Compute area of intersection
        d1 = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        d2 = d - d1
        intersection = r1 ** 2 * acos(d1 / r1) - d1 * sqrt(r1 ** 2 - d1 ** 2) \
                       + r2 ** 2 * acos(d2 / r2) - d2 * sqrt(r2 ** 2 - d2 ** 2)
        # Relative overlap is intersection / a.
        relative_overlap = intersection / a
    # Return relative overlap
    assert 0 <= relative_overlap <= 1
    return relative_overlap


def _matching_condition(blanket_feature: np.ndarray, ansatz_feature: np.ndarray, resolution: float):
    """
    Implements the matching condition.

    :param blanket_feature: Of shape (3, ).
    :param ansatz_feature: Of shape (3, ).
    :param resolution:
    :return: True if matching condition is satisfied, otherwise False.
    """
    s = blanket_feature[-1]
    s_a = ansatz_feature[-1]
    # Condition has a relative tolerance given by the constant RTOL.
    condition = ((1-RTOL) * s_a <= s <= (1 + RTOL) * max(s_a, resolution))
    return condition


def _remove_feature(features: np.ndarray, feature: np.ndarray) -> np.ndarray:
    """
    Removes a given feature from a feature array. If the feature is present repeatedly, all those occurences will be
    removed.

    :param features: Of shape (k, 3). Each row corresponds to a feature.
    :param feature: Of shape (3, ). The feature to be removed.
    :returns: The new array of shape (k-1, 3) with the feature removed.
    """
    indiced_to_keep = []
    for i in range(features.shape[0]):
        if not np.isclose(features[i], feature).all():
            indiced_to_keep.append(i)
    reduced_features = features[indiced_to_keep]
    return reduced_features



