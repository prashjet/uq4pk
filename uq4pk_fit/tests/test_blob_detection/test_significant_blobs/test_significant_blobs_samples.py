

import numpy as np

from uq4pk_fit.blob_detection import detect_significant_blobs_from_samples


alpha = 0.05
sigma_list = [1, 2, 4, 6, 8, 10, 12, 14]


def test_significant_blobs_from_samples():
    # Get test sample of images.
    sample = np.load("test_sample.npy")
    # Load reference image
    reference = np.load("reference_image.npy")
    # Run blob detection.
    blobs = detect_significant_blobs_from_samples(samples=sample, reference=reference, alpha=alpha,
                                                  sigma_list=sigma_list)

