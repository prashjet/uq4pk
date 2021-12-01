
from matplotlib import pyplot as plt
import numpy as np
from uq4pk_fit.uq_mode.detection.feature_detection import detect_features


MIN_SCALE = 1.
MAX_SCALE = 10.


def test_output_has_right_format():
    test_img = np.loadtxt("test.csv", delimiter=",")
    features = detect_features(test_img, MIN_SCALE, MAX_SCALE)
    k = features.shape[0]
    assert features.shape == (k, 3)


def test_if_no_features_are_detected_then_none_is_returned():
    # Create featureless image
    test_img = np.ones((20, 50))
    features = detect_features(test_img, MIN_SCALE, MAX_SCALE)
    assert features is None


def test_all_features_are_detected():
    # Load test image with two features.
    test_img = np.loadtxt("test.csv", delimiter=",")
    features = detect_features(test_img, MIN_SCALE, MAX_SCALE)
    assert features.shape[0] == 2   # exactly two features must be detected
    # Let's have a look.
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for feature in features:
        y, x, scale = feature
        ax.add_patch(plt.Circle((x, y), np.sqrt(2) * scale, color="lime",
                                fill=False))
    plt.show()