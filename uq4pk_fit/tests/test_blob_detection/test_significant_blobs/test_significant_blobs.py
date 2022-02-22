import numpy as np

from uq4pk_fit.visualization import plot_blobs
from uq4pk_fit.blob_detection.significant_blobs.detect_significant_blobs import _discretize_sigma, _find_blob, \
    best_blob_first
from uq4pk_fit.blob_detection.gaussian_blob import GaussianBlob

R_MIN = 0.5
R_MAX = 15
NSCALES = 16
show = False
test_img = np.random.randn(20, 20)


def test_find_blob():
    blob = GaussianBlob(x2=10, x1=10, sigma=2, log=0.)
    blob1 = GaussianBlob(x2=1, x1=1, sigma=.5, log=0.)
    blob2 = GaussianBlob(x2=8, x1=10, sigma=2., log=0)
    blob3 = GaussianBlob(x2=10, x1=9, sigma=2, log=-1.)
    blobs = best_blob_first([blob1, blob2, blob3])
    # Should match two features
    candidate = _find_blob(blob, blobs, overlap=0.5)
    blob_list = [(blob, candidate)]
    plot_blobs(image=test_img, blobs=blob_list)
    assert np.isclose(candidate.vector, np.array([9, 10, 2, 2])).all()


def test_find_blob_returns_None():
    blob = GaussianBlob(x2=1, x1=1, sigma=3,log=0.)
    blob1 = GaussianBlob(x2=10, x1=10, sigma=3, log=0.)
    blob2 = GaussianBlob(x2=5, x1=5, sigma=1, log=0)
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

