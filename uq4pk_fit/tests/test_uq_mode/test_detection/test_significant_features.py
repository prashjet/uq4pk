
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters

from uq4pk_fit.uq_mode.detection.feature_detection import detect_features
from uq4pk_fit.uq_mode.detection.significant_features import _determine_significant, _get_resolutions, _find_feature, \
    _matching_condition, _compute_overlap, _remove_feature, SignificanceTable
from uq4pk_fit.uq_mode.detection.plotty_blobby import plotty_blobby

MIN_SCALE = 1.
MAX_SCALE = 20.


def test_determine_significant():
    # Load MAP.
    map_im = np.loadtxt("data/map.csv", delimiter=",")
    # Identify MAP features
    map_features = detect_features(map_im, MIN_SCALE, MAX_SCALE, overlap=0.9)
    map_feature_list = [feature for feature in map_features]
    plotty_blobby(image=map_im, blobs=map_feature_list)
    # Make significance table.
    significance_table = SignificanceTable(map_features)
    # Load 2nd blanket.
    blanket = np.loadtxt(f"data/blanket2.csv", delimiter=",")
    # Perform feature detection to obtain blanket features
    blanket_features = detect_features(blanket, MIN_SCALE, MAX_SCALE + 1.6, overlap=0.9)
    # Correct scale
    blanket_features[:, -1] -= 0
    # Perform matching
    significance_table = _determine_significant(blanket_features, significance_table)
    # Check that there are two detected features
    #assert significance_table.n_significant == 2
    # Visualize detected features.
    plotty_blobby(image=blanket, blobs=significance_table.get_output())
    plt.show()


def test_find_feature():
    feature = np.array([10, 10, 2])
    features = np.array([[1, 1, .5], [8, 10, 2], [10, 9, 2]])
    # Should match two features
    candidate = _find_feature(feature, features, othresh=0.5)
    assert np.isclose(candidate, np.array([10, 9, 2])).all()

def test_find_feature_returns_None():
    feature = np.array([1, 1, 3])
    features = np.array([[10, 10, 3], [5, 5, 1]])
    assert _find_feature(feature, features) is None


def test_if_more_than_two_features_fit_choose_the_one_with_highest_overlap():
    feature = np.array([2, 2, 1])
    features = np.array([[3, 3, 1], [1, 1, 2]])
    candidate = _find_feature(feature, features)
    assert np.isclose(candidate, np.array([1, 1, 2])).all()



def test_relative_overlap_disjoint():
    # If circles are disjoint, overlap should be 0.
    circ1 = np.array([1, 1, 5])
    circ2 = np.array([10, 10, 1])
    overlap = _compute_overlap(circ1, circ2)
    assert overlap == 0.


def test_relative_overlap_inside():
    # If one circle is contained in the other, overlap should be 1.
    circ1 = np.array([2, 2, 1])
    circ2 = np.array([1, 1, 3])
    overlap = _compute_overlap(circ1, circ2)
    assert overlap == 1.


def test_relative_overlap_intersect():
    # Relative overlap must be between 0 and 1.
    circ1 = np.array([1, 1, 2])
    circ2 = np.array([2, 2, 1])
    # In this case, it must even be larger than 50 %, since center of smaller circle is contained in larger circle.
    overlap = _compute_overlap(circ1, circ2)
    assert 0.5 < overlap < 1


def test_matching_condition():
    ansatz_feature = np.array([4, 4, 2])
    blanket_feature = np.array([4, 4, 3])
    # This shouldn't match
    resolution = 2
    assert not _matching_condition(blanket_feature, ansatz_feature, resolution)
    # But this should:
    resolution = 3
    assert _matching_condition(blanket_feature, ansatz_feature, resolution)


def test_remove_feature():
    features = np.array([[1, 1, 2], [17, 34, 6.56], [17, 35, 6.56]])
    feature = np.array([17, 34, 6.56])
    reduced_features = _remove_feature(features, feature)
    # Check that only the second feature was removed.
    features_should_be = np.array([[1, 1, 2], [17, 35, 6.56]])
    assert np.isclose(reduced_features, features_should_be).all()


def test_get_resolutions():
    min_scale = 1.
    max_scale = 10.
    nsteps = 4
    resolutions = _get_resolutions(min_scale, max_scale, nsteps)
    # First entry must be equal to min_scale.
    assert np.isclose(resolutions[0], min_scale)
    # max_scale must lie between resolutions[-2] and resolutions[-1]
    assert resolutions[-2] <= max_scale <= resolutions[-1]

