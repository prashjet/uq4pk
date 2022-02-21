
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from skimage.feature import blob_log

from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob
from uq4pk_fit.blob_detection.detect_blobs import detect_blobs


SHOW = False    # Set True if you want to see plots.
SIGMA_MIN = 1.
SIGMA_MAX = 25.
NUM_SIGMA = 10

sigma_step = (SIGMA_MAX - SIGMA_MIN) / (NUM_SIGMA + 1)
sigma_list = [SIGMA_MIN + n * sigma_step for n in range(NUM_SIGMA + 2)]


def test_output_has_right_format():
    test_img = np.loadtxt("data/test.csv", delimiter=",")
    blobs = detect_blobs(image=test_img, sigma_list=sigma_list)
    for blob in blobs:
        assert isinstance(blob, GaussianBlob)


def test_if_no_features_are_detected_then_emptylist_is_returned():
    # Create featureless image
    test_img = np.ones((20, 50))
    blobs = detect_blobs(image=test_img, sigma_list=sigma_list, mode="reflect")
    assert len(blobs) == 0


def test_all_blobs_are_detected():
    # Load test image with two features.
    test_img = np.loadtxt("data/test.csv", delimiter=",")
    blobs = detect_blobs(image=test_img, sigma_list=sigma_list)
    assert len(blobs) == 2   # exactly two features must be detected


def test_compare_with_skimage():
    # Compare with scikit-image's implementation of blob_log.
    # Let's have a look.
    n_r = NUM_SIGMA
    overlap = 0.5
    rthresh = 0.01
    test_img = np.loadtxt("data/map.csv", delimiter=",")
    blobs = detect_blobs(image=test_img, sigma_list=sigma_list, max_overlap=overlap, rthresh=rthresh,
                         mode="constant")
    sigma_min = SIGMA_MIN
    sigma_max = SIGMA_MAX
    features_skimage = blob_log(test_img, min_sigma=sigma_min, max_sigma=sigma_max, num_sigma=n_r, overlap=overlap,
                                threshold=rthresh * test_img.max())
    # Visualize custom implementation.
    fig = plt.figure(num="detect_features", figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for blob in blobs:
        w = blob.width
        h = blob.height
        ax.add_patch(patches.Ellipse((blob.x2, blob.x1), width=w, height=h, color="lime", fill=False))
    # Visualize scikit-image's blob_log
    fig = plt.figure(num="skimage.feature.blob_log", figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for feature in features_skimage:
        y, x, sigma = feature
        ax.add_patch(plt.Circle((x, y), 2 * np.sqrt(2) * sigma, color="lime",
                                fill=False))
    if SHOW: plt.show()


def test_detect_with_ratio():
    # Compare with scikit-image's implementation of blob_log.
    # Let's have a look.
    ratio = .5
    # make sigma list with ratios.
    sigma_list_ratio = [np.array([sigma * ratio, sigma]) for sigma in sigma_list]
    overlap = 0.5
    rthresh = 0.01
    test_img = np.loadtxt("data/map.csv", delimiter=",")
    blobs = detect_blobs(test_img, sigma_list=sigma_list_ratio, max_overlap=overlap, rthresh=rthresh,
                         mode="constant")
    # Visualize custom implementation.
    fig = plt.figure(num="detect_features", figsize=(6, 2.5))
    ax = plt.axes()
    ax.imshow(test_img, cmap="gnuplot", aspect="auto")
    for blob in blobs:
        w = blob.width
        h = blob.height
        ax.add_patch(patches.Ellipse((blob.x2, blob.x1), width=w, height=h, color="lime", fill=False))
    if SHOW: plt.show()
