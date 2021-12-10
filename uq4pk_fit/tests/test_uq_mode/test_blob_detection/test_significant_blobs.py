
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.blob_detection.significant_blobs import _compute_mapped_pairs, _discretize_radius, _find_feature, \
    _compute_overlap, _remove_feature
from uq4pk_fit.uq_mode.blob_detection.detect_features import detect_features


R_MIN = 0.5
R_MAX = 15
NSCALES = 16


def test_find_feature():
    feature = np.array([2, 10, 10])
    features = np.array([[.5, 1, 1], [2, 8, 10], [2, 10, 9]])
    # Should match two features
    candidate = _find_feature(feature, features, othresh=0.5)
    assert np.isclose(candidate, np.array([2, 10, 9])).all()


def test_find_feature_returns_None():
    feature = np.array([3, 1, 1])
    features = np.array([[3, 10, 10], [1, 5, 5]])
    assert _find_feature(feature, features) is None


def test_relative_overlap_disjoint():
    # If circles are disjoint, overlap should be 0.
    circ1 = np.array([5, 1, 1])
    circ2 = np.array([1, 10, 10])
    overlap = _compute_overlap(circ1, circ2)
    assert overlap == 0.


def test_relative_overlap_inside():
    # If one circle is contained in the other, overlap should be 1.
    circ1 = np.array([1, 2, 2])
    circ2 = np.array([3, 1, 1])
    overlap = _compute_overlap(circ1, circ2)
    assert overlap == 1.


def test_relative_overlap_intersect():
    # Relative overlap must be between 0 and 1.
    circ1 = np.array([2, 1, 1])
    circ2 = np.array([1, 2, 2])
    # In this case, it must even be larger than 50 %, since center of smaller circle is contained in larger circle.
    overlap = _compute_overlap(circ1, circ2)
    assert 0.5 < overlap < 1


def test_get_resolutions():
    min_scale = 1.
    max_scale = 10.
    nsteps = 4
    resolutions = _discretize_radius(min_scale, max_scale, nsteps)
    # First entry must be equal to min_scale.
    assert np.isclose(resolutions[0], min_scale)
    assert len(resolutions) == nsteps + 2
    # max_scale must lie between resolutions[-2] and resolutions[-1]
    assert resolutions[-2] <= max_scale <= resolutions[-1]

