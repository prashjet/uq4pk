
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.blob_detection.detect_blobs.detect_blobs import detect_blobs
from uq4pk_fit.visualization import plot_significant_blobs


def test_plot_blobs():
    test_img = np.loadtxt("map.csv", delimiter=",")
    sigma_scale = np.arange(1, 16)
    sigma_list = [np.array([0.5 * sigma, sigma]) for sigma in sigma_scale]
    blobs = detect_blobs(image=test_img, sigma_list=sigma_list)
    matched_pairs = [tuple([blob, None]) for blob in blobs]

    plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    plot_significant_blobs(ax=ax, image=test_img, blobs=matched_pairs)
    plt.show()