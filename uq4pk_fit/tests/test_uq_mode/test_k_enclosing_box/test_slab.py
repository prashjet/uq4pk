
import numpy as np
from uq4pk_fit.uq_mode.k_enclosing_box.slab import Slab, divide_slab

n = 1000


def test_slab():
    n1 = 100
    n2 = 200
    points = np.random.randn(n, 2)
    y_values = points[:, 1]
    y_sorted = np.sort(y_values)
    slab = Slab(y_min=y_sorted[n1], y_max=y_sorted[n2], points=points)
    assert slab.num_in_slab == n2 - n1
    mask_below = slab.points_below(points)
    points_below = points[mask_below]
    assert points_below.shape[0] == n1 + 1
    mask_above = slab.points_above(points)
    points_above = points[mask_above]
    assert points_above.shape[0] == n - n2 - 1

def test_slab_half():
    n1 = 100
    n2 = 200
    points = np.random.randn(n, 2)
    y_values = points[:, 1]
    y_sorted = np.sort(y_values)
    slab = Slab(y_min=y_sorted[n1], y_max=y_sorted[n2], points=points)
    y_half = slab.y_half
    points_in_slab = slab._y_points_in_slab()
    num_in_lower_half = points_in_slab[points_in_slab <= y_half].size
    assert num_in_lower_half == int((n2 - n1) / 2)

def test_divide_slab():
    n1 = 100
    n2 = 200
    points = np.random.randn(n, 2)
    y_values = points[:, 1]
    y_sorted = np.sort(y_values)
    slab = Slab(y_min=y_sorted[n1], y_max=y_sorted[n2], points=points)
    slab1, slab2 = divide_slab(slab)
    assert slab1.num_in_slab == slab2.num_in_slab == (n2 - n1) / 2