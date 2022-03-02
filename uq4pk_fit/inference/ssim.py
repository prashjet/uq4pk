import numpy as np
from skimage.metrics import structural_similarity


def ssim(image: np.ndarray, reference: np.ndarray) -> float:
    """
    Computes the structural similarity of an image to a reference image.
    """
    # Check that the images have the same shape
    assert image.ndim == 2
    assert image.shape == reference.shape

    # Compute structural similarity
    similiarity = structural_similarity(image, reference, data_range=reference.max() - reference.min())

    return similiarity