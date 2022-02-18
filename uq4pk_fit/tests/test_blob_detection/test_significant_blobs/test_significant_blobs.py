import numpy as np

from uq4pk_fit.blob_detection.significant_blobs.compute_significant_blobs import _discretize_sigma, _find_blob
from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob

R_MIN = 0.5
R_MAX = 15
NSCALES = 16


def test_find_blob():
    blob = GaussianBlob(x=10, y=10, sigma_x=2, sigma_y=2, log=0.)
    blob1 = GaussianBlob(x=1, y=1, sigma_x=.5, sigma_y=.5, log=0.)
    blob2 = GaussianBlob(x=8, y=10, sigma_x=2., sigma_y=2., log=0)
    blob3 = GaussianBlob(x=10, y=9, sigma_x=2, sigma_y=2, log=0)
    blobs = [blob1, blob2, blob3]
    # Should match two features
    candidate = _find_blob(blob, blobs, overlap=0.5)
    assert np.isclose(candidate.vector, np.array([8, 10, 2, 2])).all()


def test_find_blob_returns_None():
    blob = GaussianBlob(x=1, y=1, sigma_x=3, sigma_y=3, log=0.)
    blob1 = GaussianBlob(x=10, y=10, sigma_x=3, sigma_y=3, log=0.)
    blob2 = GaussianBlob(x=5, y=5, sigma_x=1, sigma_y=1, log=0)
    blobs = [blob1, blob2]
    assert _find_blob(blob, blobs, overlap=0.5) is None


def test_get_resolutions():
    min_sigma = 1.
    max_sigma = 10.
    nsteps = 4
    resolutions = _discretize_sigma(min_sigma, max_sigma, nsteps)
    # First entry must be equal to min_scale.
    assert np.isclose(resolutions[0], min_sigma)
    assert len(resolutions) == nsteps + 2
    # max_scale must lie between resolutions[-2] and resolutions[-1]
    assert resolutions[-2] <= max_sigma <= resolutions[-1]

