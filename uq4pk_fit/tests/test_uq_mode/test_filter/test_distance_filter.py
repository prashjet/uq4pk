
import numpy as np

from uq4pk_fit.uq_mode.filter.distance_filter import DistanceFilter


def test_dist():
    m = 10
    n = 10
    a = 2
    b = 2
    position = np.array([a, b])
    h = 2.
    def weighting(d):
        return d
    distance_filter = DistanceFilter(m=m, n=n, a=a, b=b, position=position, weighting=weighting, scaling=h)
    x1 = np.array([0., 0.])
    x2 = np.array([3., 2.])
    dist_ref = np.linalg.norm(x1 - x2) / h
    assert np.isclose(dist_ref, distance_filter._dist(x1, x2))

