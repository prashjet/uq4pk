
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.detection.scale_space_representation import scale_space_representation


def test_scale_space_representation():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    nscales = 6
    # Compute scale space representation
    scales = [0.3 * 1.6 ** k for k in range(nscales)]
    f_ssr = scale_space_representation(f, scales)
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(scales), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(nscales - 1):
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
    nscales = 6
    # Compute scale space representation
    scales = [0.3 * 1.6 ** k for k in range(nscales)]
    f_ssr = scale_space_representation(f, scales, mode="reflect")
    # Check that scale-space representation has correct dimension
    m, n = f.shape
    assert f_ssr.shape == (len(scales), m, n)
    # Check non-enhancement property of maxima, i.e. the value of the maxima must decrease with increasing scale.
    for i in range(nscales - 1):
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