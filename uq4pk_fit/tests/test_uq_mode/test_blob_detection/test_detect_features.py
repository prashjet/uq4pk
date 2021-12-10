
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from skimage.feature import blob_log

from uq4pk_fit.uq_mode.blob_detection.detect_features import detect_features


SIGMA_MIN = 1.
SIGMA_MAX = 25.
NUM_SIGMA = 10


def test_output_has_right_format():
    test_img = np.loadtxt("data/test.csv", delimiter=",")
    features = detect_features(test_img, SIGMA_MIN, SIGMA_MAX)
    k = features.shape[0]
    assert features.shape == (k, 3)


def test_if_no_features_are_detected_then_none_is_returned():
    # Create featureless image
    test_img = np.ones((20, 50))
    features = detect_features(test_img, SIGMA_MIN, SIGMA_MAX, mode="reflect")
    assert features is None


def test_all_features_are_detected():
    # Load test image with two features.
    test_img = np.loadtxt("data/test.csv", delimiter=",")
    features = detect_features(test_img, SIGMA_MIN, SIGMA_MAX)
    assert features.shape[0] == 2   # exactly two features must be detected


def test_compare_with_skimage():
    # Compare with scikit-image's implementation of blob_log.
    # Let's have a look.
    n_r = NUM_SIGMA
    overlap = 0.5
    rthresh = 0.01
    test_img = np.loadtxt("data/map.csv", delimiter=",")
    features = detect_features(test_img, SIGMA_MIN, SIGMA_MAX, num_sigma=n_r, rthresh=rthresh, overlap=overlap,
                               mode="constant")
    sigma_min = SIGMA_MIN
    sigma_max = SIGMA_MAX
    features_skimage = blob_log(test_img, min_sigma=sigma_min, max_sigma=sigma_max, num_sigma=n_r, overlap=overlap,
                                threshold=rthresh * test_img.max())
    # Visualize custom implementation.
    fig = plt.figure(num="detect_features", figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for feature in features:
        s_x, s_y, y, x = feature
        w = 4 * np.sqrt(2) * s_x
        h = 4 * np.sqrt(2) * s_y
        ax.add_patch(patches.Ellipse((x, y), width=w, height=h, color="lime",
                                fill=False))

    # Visualize scikit-image's blob_log
    fig = plt.figure(num="skimage.feature.blob_log", figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for feature in features_skimage:
        y, x, sigma = feature
        ax.add_patch(plt.Circle((x, y), 2 * np.sqrt(2) * sigma, color="lime",
                                fill=False))
    plt.show()


def test_detect_with_ratio():
    # Compare with scikit-image's implementation of blob_log.
    # Let's have a look.
    ratio = .5
    n_r = NUM_SIGMA
    overlap = 0.5
    rthresh = 0.05
    test_img = np.loadtxt("data/map.csv", delimiter=",")
    features = detect_features(test_img, SIGMA_MIN, SIGMA_MAX, num_sigma=n_r, rthresh=rthresh, overlap=overlap,
                               mode="constant", ratio=ratio)
    # Visualize custom implementation.
    fig = plt.figure(num="detect_features", figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for feature in features:
        s_x, s_y, y, x = feature
        w = 2 * np.sqrt(2) * s_x
        h = 2 * np.sqrt(2) * s_y
        ax.add_patch(patches.Ellipse((x, y), width=w, height=h, color="lime", fill=False))
    plt.show()
