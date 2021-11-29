
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.detection.feature_detection import blob_dog
from uq4pk_fit.uq_mode.detection.significant_features import _determine_resolution, _map_features, \
    _matching_condition, _compute_overlap, _remove_feature, OTHRESH
from uq4pk_fit.uq_mode.detection.blob_plot import plot_blob

MIN_SCALE = 1.
MAX_SCALE = 10.


def test_determine_resolution():
    # Load MAP.
    map_im = np.loadtxt("data/map.csv", delimiter=",")
    # Identify MAP features
    map_features = blob_dog(map_im, MIN_SCALE, MAX_SCALE)
    # Load blanket-stack.
    features_of_blanket_list = []
    resolution_list = [MIN_SCALE * 1.6 ** i for i in range(5)]
    for i in range(1, 6):
        # Load i-th blanket.
        blanket_i = np.loadtxt(f"data/blanket{i}.csv", delimiter=",")
        # Perform feature detection to obtain list of features.
        features_i = blob_dog(blanket_i, MIN_SCALE, MAX_SCALE)
        features_of_blanket_list.append(features_i)
    # Perform matching
    sig_features = _determine_resolution(ansatz_features=map_features,
                                         list_of_blanket_features=features_of_blanket_list,
                                         resolution_list=resolution_list)
    # Visualize detected features and also features in MAP for comparison.
    plot_blob(image=map_im, blobs=map_features)
    plot_blob(image=map_im, blobs=sig_features)
    plt.show()


def test_map_features():
    blanket_features = np.array([[1, 1, 1], [10, 10, 2]])
    ansatz_features = np.array([[1, 1, .5], [8, 10, 2], [10, 9, 2]])
    # Should match two features
    mapped_pairs = _map_features(blanket_features, ansatz_features)
    assert len(mapped_pairs) == 2
    # Also, each feature should have the right overlap
    for feature1, feature2 in mapped_pairs:
        overlap = _compute_overlap(feature1, feature2)
        assert overlap >= OTHRESH


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

