

from typing import List, Sequence, Tuple, Union
import miniball
import numpy as np

from ..detect_blobs import detect_blobs
from uq4pk_fit.gaussian_blob.gaussian_blob import GaussianBlob
from .detect_significant_blobs import _find_blob

SigmaList = Sequence[Union[float, np.ndarray]]


def detect_significant_blobs_from_samples(samples: np.ndarray, sigma_list: SigmaList, reference: np.ndarray,
                                          alpha: float = 0.05, rthresh: float = 0.05, overlap: float = 0.5) \
        -> List[Tuple[GaussianBlob, Union[GaussianBlob, None]]]:
    """
    Performs uncertainty-aware blob detection with automatic scale selection.

    :param samples: Array of shape (s, n1, n2), corresponding to the s-sampled images.
    :param sigma_list: The list of standard deviations used for the FCIs.
    :param reference: The image for which significant features need to be determined, e.g. the MAP estimate.
    :param alpha: The credibility parameter.
    :param rthresh: The relative threshold for filtering out weak features.
    :param overlap: The maximum allowed overlap for detected features.
    :return: A list of tuples. Each tuple is of the form (blob1, blob2), where blob1 is a GaussianBlob object
        corresponding to a detected blob in the reference image, and blob2 is either None, meaning that the blob
        is not significant, or another GaussianBlob object representing the significant blob associated to blob1.
    """
    # Check input for consistency.
    assert samples.ndim == 3
    s, n1, n2 = samples.shape
    assert reference.shape == (n1, n2)
    assert 0 <= rthresh <= 1.
    assert 0 < overlap <= 1.

    # Identify features in reference image.
    reference_blobs = detect_blobs(image=reference, sigma_list=sigma_list, max_overlap=overlap, rthresh=rthresh,
                                   mode="constant")

    mapped_pairs = _match_to_samples(blobs=reference_blobs, samples=samples, alpha=alpha, rthresh=rthresh,
                                     overlap=overlap)

    return mapped_pairs


def _match_to_samples(blobs: Sequence[GaussianBlob], samples: np.ndarray, sigma_list: SigmaList, alpha: float,
                      rthresh: float, overlap: float) -> List[Tuple[GaussianBlob, Union[GaussianBlob, None]]]:
    """
    Determines (1-\alpha)-significant blobs from samples.

    :param blobs: The blobs detected in the reference image.
    :param samples:
    :param rthresh:
    :param overlap:
    :return:
    """

    # Perform blob detection for each sample. This results in a list of lists `sample_blobs`, where each inner list
    # contains the blobs that where detected in given sample image.
    n_samples = samples.shape[0]
    sample_blobs = []
    for image in samples:
        blobs_of_sample = detect_blobs(image=image, sigma_list=sigma_list, rthresh=rthresh, max_overlap=overlap)
        sample_blobs.append(blobs_of_sample)
    # For each reference blob, determine the matching sample blobs.
    matched_pairs = []
    for blob in blobs:
        matching_blobs = []
        for blob_list in sample_blobs:
            matching_blob = _find_blob(blob, blob_list, overlap=overlap)
            if matching_blob is not None:
                matching_blobs.append(matching_blob)
        # Compute the ratio [number_of_matching_blobs]/[n_samples].
        ratio = len(matching_blobs) / n_samples
        if ratio >= 1 - alpha:
            # If the ratio is above 1 - alpha, the blob is significant and we have to determine the `significant_blob`
            # object. The significant blob is defined as the one that covers 95% of all sample-blobs.
            significant_blob = _find_covering_blob(reference_blob=blob, blobs=matching_blobs, alpha=alpha)
        else:
            # If the ratio is below 1 - alpha, the blob is not significant.
            significant_blob = None
        matched_pairs.append((blob, significant_blob))

    # We return the list of blob-significant_blob pairs.
    return matched_pairs


def _find_covering_blob(reference_blob: GaussianBlob, blobs: Sequence[GaussianBlob], alpha: float) -> GaussianBlob:
    """
    Finds smallest blob that covers 1-alpha of the given blobs. The implementation is inspired by
    https://stackoverflow.com/questions/39705456/smallest-bounding-sphere-containing-x-of-points and Jack Ritter's
    answer to
    https://stackoverflow.com/questions/9063453/how-to-compute-the-smallest-bounding-sphere-enclosing-other-bounding-spheres

    :param blobs: List of GaussianBlob objects.
    :param alpha: Significance parameter.
    :return: An object of type Gaussian blob.
    """
    # Compute the geometric median of the blob centers.
    # Only keep the 1 - alpha blobs closest to reference blob.
    # For each blob, get the 4 x-y min/max points. Store all these points in an array `point_bag`
    point_bag = ...
    # Compute the smallest enclosing circle for point_bag (rows correspond to points, columns to coordinates).
    center, radius = miniball.get_bounding_ball(S=point_bag)
    # Represent the circle as a GaussianBlob.
    covering_blob = GaussianBlob(x1=center[0], x2=center[1], log=0., sigma=radius)

    return covering_blob





