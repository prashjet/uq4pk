
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteGradient
from uq4pk_fit.uq_mode.detection.blob_operator import blob_operator


def test_blob_operator():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    nscales = 6
    # Define scale discretization.
    scales = [0.3 * 1.6 ** k for k in range(nscales)]
    # Compute blob operator
    m, n = f.shape
    bob = blob_operator(scales=scales, m=m, n=n)
    # Apply to test image and visualize output
    g = bob @ f.flatten()
    g3d = np.reshape(g, (len(scales), m, n))
    i = 0
    for g_h in g3d:
        fig = plt.figure(num=f"h = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(g_h, cmap="gnuplot")
        i += 1
    plt.show()


def test_shape_operator():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    nscales = 6
    # Define scale discretization.
    scales = [0.3 * 1.6 ** k for k in range(nscales)]
    # Compute blob operator
    m, n = f.shape
    bob = blob_operator(scales=scales, m=m, n=n)
    gradient3d = DiscreteGradient(shape=(len(scales), m, n)).mat
    # Apply to test image and visualize output
    g = gradient3d @ bob @ f.flatten()
    g3d = np.reshape(g, (3 * len(scales), m, n))
    # Sum up
    g_list = np.array_split(g3d, 3)
    gsum = sum(g for g in g_list)
    i = 0
    for g_h in gsum:
        fig = plt.figure(num=f"h = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(g_h, cmap="gnuplot")
        i += 1
    plt.show()
