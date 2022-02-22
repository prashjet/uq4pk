
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.blob_detection.minimize_blobiness.blob_operator import blob_operator


SHOW = False    # Set True if you want to see plots.


def test_blob_operator():
    # Load test image.
    f = np.loadtxt("data/map.csv", delimiter=",")
    num_sigma = 10
    sigma_min = 1
    sigma_max = 20
    sigma_step = (sigma_max - sigma_min) / (num_sigma + 1)
    # Define scale discretization.
    sigmas = [sigma_min + n * sigma_step for n in range(num_sigma + 2)]
    scales = [0.5 * sigma ** 2 for sigma in sigmas]
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
    if SHOW: plt.show()
