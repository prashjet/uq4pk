
from math import acos, pi, sqrt
import numpy as np
from typing import List, Union

from ..fci import fci
from ..filter import SquaredExponentialFilterFunction
from ..linear_model import LinearModel
from .feature_detection import detect_features
from .span_blanket import span_blanket


RTOL = 0.05     # Relative tolerance for matching condition.
OTHRESH = 0.5   # Relative overlap-threshold for feature mapping. Features are mapped if overlap is larger than this.


# Create significance_table data type
class SignificanceTable:

    def __init__(self, features: np.ndarray):
        # Check input.
        assert features.shape[1] == 3
        self._features = [feature for feature in features]  # Convert to list.
        self._size = len(features)
        # Initialize significance list
        self._significant_features = [None] * self._size

    def get_feature(self, i: int) -> np.ndarray:
        return self._features[i]

    def get_output(self) -> Union[List[np.ndarray], None]:
        """
        Returns all significant features as array of shape (k, 3) or None, if no significant features exist.
        """
        # If no significant features exist, return None.
        if self.n_significant == 0:
            return None
        else:
            # Create the return objects of shape (3, 2).
            output = []
            for map_feature, significant_feature in zip(self._features, self._significant_features):
                if significant_feature is not None:
                    feature_arr = np.vstack([map_feature, significant_feature])
                    output.append(feature_arr)
            return output

    def update(self, i: int, feature: np.ndarray):
        # Check that input has right format.
        assert feature is None or feature.shape == (3, )
        significant_feature_i = self._significant_features[i]
        if significant_feature_i is not None and feature is not None:
            # Finally, if significance is not None, then check which one has the smaller s_b - r and choose that one.
            if significant_feature_i[2] > feature[2]:
                self._significant_features[i] = feature
        else:
            self._significant_features[i] = feature

    @property
    def size(self) -> int:
        return self._size

    @property
    def n_significant(self) -> int:
        """
        The number of significant features.
        """
        n = 0
        for i in range(self._size):
            if self._significant_features[i] is not None:
                n += 1
        return n


def significant_features(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, minscale: float,
                         maxscale: float, minres: float = None, maxres: float = None, nsteps: int = 10) \
        -> Union[List[np.ndarray], None]:
    """
    Performs uncertainty-aware blob detection with automatic scale selection.

    Performs feature matching to determine which features in ``ansatz`` are significant with respect to
    ``blanket_list``, and at which resolution. For this, we note the following:
    - A feature not present in ``ansatz`` cannot be significant.
    - Also, the size and position of the possible significant features are determined by ``ansatz``.
    - Hence, we only have to determine which features are significant, and the resolution of the signficant features.

    This is implemented by iterating over all blankets, starting at the highest resolution (note that high resolution
    corresponds to low 'r').
    1. Perform feature detection on the current blanket. Only features at scales greater or equal
        r will be detected.
    2. Map the detected features in the current blanket to the features in ``ansatz``. Then, all features that do not
        match any feature in ``ansatz`` are dropped.
    3. For each mapped feature pair, we compare the scales. Let 's_b' be the scale at which the feature was detected in
        the current blanket, and 's_a' be the scale at which it was detected in the MAP estimate. We then store the
        blanket feature (i_b, j_b, s_b - r)
        Theoretically, there always should hold:
                    s_b - r >= s_a.
        The reason for this is that if we filter the MAP estimate with a Gaussian of scale r, the feature of scale s_a
        will become a feature of scale s_a + r. However, if that feature is also present in the r-blanket, it must be
        at a scale s_b >= r + s_a, since the r-blanket is by definition feature-minimizing.
    After iterating over all blankets, we pick for each significant feature the blanket value (i_b, j_b, s_b - r)
    that minimizes s_b - r.

    :param alpha: The credibility parameter.
    :param m: Number of image rows.
    :param n: Number of image columns.
    :param model: The statistical mode. Must be a :py:class:`LinearModel` object.
    :param x_map: The MAP estimate.
    :param minscale: The minimal scale at which features should be detected.
    :param maxscale: The maximal scale at which features should be detected.
    :param minres: Minimal resolution at which features should be detected.
    :param maxres: Maximal resolution at which features should be detected.
    :param nsteps: Number of resolution steps. For example, if ``minscale = 1``, ``maxscale = 10`` and ``nsteps = 4``,
        the resolutions 1, 2.5, 5, 7.5 and 10 are checked.
    :returns: A list of significant features is returned. Each element is an array of shape (2, 3), where the first
        row corresponds to the MAP feature, and the second row corresponds to the significant feature.
    """
    # Check input for consistency.
    _check_input_significant_features(alpha, m, n, model, x_map, minscale, maxscale)
    # If minres and maxres are not given, they default to minscale and maxscale
    if minres is None:
        minres = minscale / 1.6
    if maxres is None:
        maxres = maxscale * 1.6
    # Identify features in MAP estimate
    map_im = np.reshape(x_map, (m, n))
    map_features = detect_features(map_im, minscale, maxscale, overlap=OTHRESH)
    # Select resolutions between minres and maxres.
    resolutions = _get_resolutions(minres, maxres, nsteps)
    # Initialize significance table.
    significance_table = SignificanceTable(map_features)
    # Iterate over all scales.
    for resolution in resolutions:
        print(f"RESOLUTION {resolution}")
        # Compute blanket at given scale.
        blanket = _compute_blanket(alpha, m, n, model, x_map, resolution)
        # Compute features of blanket (features cannot be detected at scale < resolution).
        blanket_features = detect_features(blanket, resolution, maxscale + resolution)
        # Subtract r from the last entries, so that each feature is of the form (i_b, j_b, s_b - r).
        blanket_features[:, 2] -= resolution
        # Determine significant features at current resolution and remove matched features from ``map_features``.
        significance_table = _determine_significant(blanket_features, significance_table)
        # The significance table is now a list of tuples of the form (map_feature, blanket_feature),
        # where 'blanket_feature' is None if no corresponding blanket_feature was found, else it is the entry
        # (i_b, j_b, s_b - r).
    # Return the significance features.
    return significance_table.get_output()


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


def _get_resolutions(min_scale: float, max_scale: float, nsteps: int) -> List[float]:
    """
    Returns list of resolutions of a constant ratio, such that the range [min_scale, max_scale] is covered.

    :param min_scale: Minimum resolution.
    :param max_scale: Maximum resolution.
    :return: The list of resolutions.
    """
    # Check the input for sensibleness.
    assert min_scale <= max_scale
    assert nsteps >= 1
    step_size = (max_scale - min_scale) / nsteps
    resolutions = [min_scale + n * step_size for n in range(nsteps + 1)]
    return resolutions


def _compute_blanket(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, scale: float)\
        -> np.ndarray:
    """
    Compute blankets at given resolution for the given model.

    :param alpha:
    :param m:
    :param n:
    :param model:
    :param x_map:
    :param scale: The scale at which the blanket should be computed.
    :returns: Of shape (m, n). The computed blanket
    """
    # Setup Gaussian filter function.
    scale_int = max(np.ceil(scale).astype(int), 3)
    filter_function = SquaredExponentialFilterFunction(m=m, n=n, a=1, b=1, c=2 * scale_int, d=2 * scale_int,
                                                       h=scale)
    # Compute filtered credible interval.
    ci = fci(alpha=alpha, model=model, x_map=x_map, ffunction=filter_function)
    # Compute minimally bumpy element using taut_string
    lower = np.reshape(ci[:, 0], (m, n))
    upper = np.reshape(ci[:, 1], (m, n))
    blanket = span_blanket(lb=lower, ub=upper)
    # Assert that blanket has the right format before returning it.
    assert blanket.shape == (m, n)
    return blanket


def _determine_significant(blanket_features: np.ndarray, significance_table: SignificanceTable) \
        -> SignificanceTable:
    """
    For a given blanket-feature array, determines whether any features "match" the features in ``map_features``
    at the given resolution.

    :param blanket_features: Of shape (k, 3).
    :param significance_table:
    :returns: The updated significance table.
    """
    # Iterate over the significance_table.
    for i in range(significance_table.size):
        map_feature_i = significance_table.get_feature(i)
        # Find the feature matching the map_feature
        blanket_feature = _find_feature(map_feature_i, blanket_features, othresh=OTHRESH)
        # If a match was found, remove the matched feature from blanket_features.
        if blanket_feature is not None:
            blanket_features = _remove_feature(blanket_features, blanket_feature)
        # Update significance table.
        significance_table.update(i, blanket_feature)
    return significance_table


def _find_feature(feature: np.ndarray, features: np.ndarray, othresh: float = 0.5) -> Union[np.ndarray, None]:
    """
    Find a feature in a given collection of features.
    A feature is mapped if the overlap to another feature is more than a given threshold.
    If there is more than one matching feature, the one with the largest relative overlap is selected.
    If the relative overlap is 1 for more than one feature in ``features``, the first matching feature is selected.

    :param feature: Of shape (3, ). The feature to be found.
    :param features: Of shape (n, 3). The array of features in which ``feature`` is searched.
    :return: The mapped feature. If no fitting feature is found, None is returned.
    """
    # Check that input has right format.
    assert feature.shape == (3, )
    assert features.shape[1] == 3
    # Iterate over features.
    matching_features = []
    for candidate in features:
        overlap = _compute_overlap(feature, candidate)
        if overlap >= othresh:
            candidate_with_overlap = np.append(candidate, overlap)
            matching_features.append(candidate_with_overlap)
    # If no matching feature was found, return None
    if len(matching_features) == 0:
        found = None
    else:
        matching_features = np.array(matching_features)
        # Otherwise, choose the feature with maximal overlap.
        maximizing_index = np.argmax(matching_features[:, -1])
        found = matching_features[maximizing_index, :-1]
    # Check that output has right format
    assert found is None or found.shape == (3, )
    return found


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
