
import numpy as np

from uq4pk_fit.visualization import plot_blobs
from uq4pk_fit.blob_detection import GaussianBlob
from uq4pk_fit.blob_detection.blob_geometry import compute_overlap

SHOW = False
test_image = np.loadtxt("data/test.csv", delimiter=",")


def test_blobs_not_intersecting():
    blob1 = GaussianBlob(x1=4, x2=51, sigma=np.array([.5, 1.]), log=0.)
    blob2 = GaussianBlob(x1=11, x2=52, sigma=np.array([4., 8.]), log=-1.)
    overlap = compute_overlap(blob1, blob2)
    print(f"overlap = {overlap}")
    if SHOW: plot_blobs(image=test_image, blobs=[(blob1, blob2)])
    assert np.isclose(overlap, 0.)


def test_blob_inside_other_blob():
    blob1 = GaussianBlob(x1=5, x2=6, sigma=np.array([1., 1.]), log=0.)
    blob2 = GaussianBlob(x1=7, x2=7, sigma=np.array([3., 5.]), log=-1.)
    overlap = compute_overlap(blob1, blob2)
    print(f"overlap = {overlap}")
    if SHOW: plot_blobs(image=test_image, blobs=[(blob1, blob2)])
    assert np.isclose(overlap, 1.)


def test_blobs_slightly_intersect():
    blob1 = GaussianBlob(x1=5, x2=8, sigma=np.array([1, 2]), log=0.)
    blob2 = GaussianBlob(x1=8, x2=9, sigma=np.array([2, 4]), log=-1.)
    overlap = compute_overlap(blob1, blob2)
    print(f"overlap = {overlap}")
    if SHOW: plot_blobs(image=test_image, blobs=[(blob1, blob2)])
    assert 0.3 < overlap < .4
