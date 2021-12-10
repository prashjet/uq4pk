
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.blob_detection.scale_space_representation import scale_space_representation


NSCALES = 16
R_MIN = 0.5
R_MAX = 25


def test_scale_space_representation():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    # Compute scale space representation
    r_step = (R_MAX - R_MIN) / NSCALES
    radii = [R_MIN + n * r_step for n in range(NSCALES)]
    scales = [0.25 * r ** 2 for r in radii]
    f_ssr = scale_space_representation(f, scales)
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(scales), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(NSCALES - 1):
        max_0 = f_ssr[i].max()
        max_1 = f_ssr[i + 1].max()
        assert max_0 >= max_1
    # Finally, plot the scale-slices.
    i = 0
    for f_h in f_ssr:
        fig = plt.figure(num=f"h = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    plt.show()


def test_reflect():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    r_step = (R_MAX - R_MIN) / NSCALES
    radii = [R_MIN + n * r_step for n in range(NSCALES)]
    scales = [0.25 * r ** 2 for r in radii]
    f_ssr = scale_space_representation(f, scales, mode="reflect")
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(scales), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(NSCALES - 1):
        max_0 = f_ssr[i].max()
        max_1 = f_ssr[i + 1].max()
        assert max_0 >= max_1
    # Finally, plot the scale-slices.
    i = 0
    for f_h in f_ssr:
        fig = plt.figure(num=f"h = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    plt.show()


def test_with_ratio():
    ratio = 0.5
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    r_step = (R_MAX - R_MIN) / NSCALES
    radii = [R_MIN + n * r_step for n in range(NSCALES)]
    scales = [0.25 * r ** 2 for r in radii]
    f_ssr = scale_space_representation(f, scales, ratio=ratio, mode="constant")
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(scales), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(NSCALES - 1):
        max_0 = f_ssr[i].max()
        max_1 = f_ssr[i + 1].max()
        assert max_0 >= max_1
    # Finally, plot the scale-slices.
    i = 0
    for f_h in f_ssr:
        fig = plt.figure(num=f"h = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    plt.show()