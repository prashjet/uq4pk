
import numpy as np

from uq4pk_fit.uq_mode.k_enclosing_box.s_structure import Structure1D
from uq4pk_fit.uq_mode.k_enclosing_box.slab import Slab


n = 1000
n1 = 100
n2 = 200
n3 = 900
n4 = 1000
k = 900


def test_init():
    points = np.random.randn(n, 2)
    marked = np.full((n,), True)
    s = Structure1D(points=points, marked=marked, k=k)
    assert s.enough_points


def test_delete():
    points = np.random.randn(n, 2)
    y_values = points[:, 1]
    y_sorted = np.sort(y_values)
    y_min_tau = y_sorted[n1]
    y_max_tau = y_sorted[n2]
    y_min_sigma = y_sorted[n3]
    y_max_sigma = y_sorted[n4-1]
    tau = Slab(points=points, y_min=y_min_tau, y_max=y_max_tau)
    sigma = Slab(points=points, y_min=y_min_sigma, y_max=y_max_sigma)
    marked = np.full((n,), True)
    s = Structure1D(points=points, marked=marked, k=k)
    s.delete(sigma, tau)
    assert s._points.shape[0] < n


def test_unmark():
    points = np.random.randn(n, 2)
    y_values = points[:, 1]
    y_sorted = np.sort(y_values)
    y_min_tau = y_sorted[n1]
    y_max_tau = y_sorted[n2]
    y_min_sigma = y_sorted[n3]
    y_max_sigma = y_sorted[n4 - 1]
    tau = Slab(points=points, y_min=y_min_tau, y_max=y_max_tau)
    sigma = Slab(points=points, y_min=y_min_sigma, y_max=y_max_sigma)
    marked = np.full((n,), True)
    s = Structure1D(points=points, marked=marked, k=k)
    s.unmark(sigma, tau)
    n_marked = np.sum(s._marked)
    assert n_marked != n