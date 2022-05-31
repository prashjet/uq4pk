
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.gaussian_blob.scale_space_representation.scale_space_representation import scale_space_representation


SHOW = True    # Set True if you want to see plots.
NSCALES = 16
SIGMA_MIN = 0.5
SIGMA_MAX = 25


def test_scale_space_representation():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    # Compute scale space representation
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / NSCALES
    radii = [SIGMA_MIN + n * sigma_step for n in range(NSCALES)]
    scales = [0.5 * r ** 2 for r in radii]
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
        fig = plt.figure(num=f"t = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    if SHOW: plt.show()


def test_reflect():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / NSCALES
    radii = [SIGMA_MIN + n * sigma_step for n in range(NSCALES)]
    scales = [0.5 * r ** 2 for r in radii]
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
        fig = plt.figure(num=f"t = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    if SHOW: plt.show()


def test_with_ratio():
    ratio = 0.5
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / NSCALES
    sigmas = [SIGMA_MIN + n * sigma_step for n in range(NSCALES)]
    scales = [0.5 * np.array([s * ratio, s]) ** 2 for s in sigmas]
    f_ssr = scale_space_representation(f, scales, mode="constant")
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
        fig = plt.figure(num=f"t = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    if SHOW: plt.show()