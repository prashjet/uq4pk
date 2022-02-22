
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.discretization import Window, LocalizationWindows


m = 12
n = 53
w1 = 2
w2 = 3
im_ref = np.random.randn(m, n)


def test_create_window():
    center = np.array([5, 20])
    window = Window(im_ref=im_ref, center=center, w1=w1, w2=w2)
    assert window.dim == m * n
    assert window.m == m
    assert window.n == n
    assert window.dof == (2 * w1 + 1) * (2 * w2 + 1)


def test_window_at_boundary():
    center = np.array([11, 52])
    window = Window(im_ref=im_ref, center=center, w1=w1, w2=w2)
    assert window.dof == (w1 + 1) * (w2 + 1)


def test_create_localization_windows():
    localization_windows = LocalizationWindows(im_ref=im_ref, w1=w1, w2=w2)
    assert localization_windows.dim == m * n

