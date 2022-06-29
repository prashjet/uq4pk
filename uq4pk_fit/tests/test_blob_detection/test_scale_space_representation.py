
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.gaussian_blob import scale_space_representation


SHOW = True    # Set True if you want to see plots.
NSCALES = 16
SIGMA_MIN = 0.5
SIGMA_MAX = 25


def test_scale_space_representation():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    # Compute scale space representation
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / NSCALES
    sigmas = [np.ones(2, ) *(SIGMA_MIN + n * sigma_step) for n in range(NSCALES)]
    f_ssr = scale_space_representation(f, sigmas)
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(sigmas), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(NSCALES - 1):
        max_0 = f_ssr[i].max()
        max_1 = f_ssr[i + 1].max()
        assert max_0 >= max_1
    # Finally, plot the scale-slices.
    i = 0
    for f_h in f_ssr:
        fig = plt.figure(num=f"sigma = {sigmas[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    if SHOW: plt.show()


def test_reflect():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / NSCALES
    sigmas = [np.ones(2, ) *(SIGMA_MIN + n * sigma_step) for n in range(NSCALES)]
    f_ssr = scale_space_representation(f, sigmas, mode="reflect")
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(sigmas), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(NSCALES - 1):
        max_0 = f_ssr[i].max()
        max_1 = f_ssr[i + 1].max()
        assert max_0 >= max_1
    # Finally, plot the scale-slices.
    i = 0
    for f_h in f_ssr:
        fig = plt.figure(num=f"t = {sigmas[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    if SHOW: plt.show()


def test_with_ratio():
    ratio = 0.5
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    sigma_step = (SIGMA_MAX - SIGMA_MIN) / NSCALES
    sigmas = [SIGMA_MIN + n * sigma_step for n in range(NSCALES)]
    sigma_list = [np.array([ratio * sigma, sigma]) for sigma in sigmas]
    f_ssr = scale_space_representation(f, sigma_list)
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(sigma_list), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(NSCALES - 1):
        max_0 = f_ssr[i].max()
        max_1 = f_ssr[i + 1].max()
        assert max_0 >= max_1
    # Finally, plot the scale-slices.
    i = 0
    for f_h in f_ssr:
        fig = plt.figure(num=f"t = {sigma_list[i]}", figsize=(6, 2.5))
        plt.imshow(f_h, cmap="gnuplot")
        i += 1
    if SHOW: plt.show()