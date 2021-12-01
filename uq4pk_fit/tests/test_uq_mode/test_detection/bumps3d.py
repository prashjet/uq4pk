
from matplotlib import pyplot as plt
import numpy as np
import os
import skimage.morphology as morphology

from uq4pk_fit.special_operators import NormalizedLaplacian
from uq4pk_fit.uq_mode.detection.minimize_bumps import minimize_bumps
from uq4pk_fit.uq_mode.detection.plotty_blobby import plotty_blobby

LOC = "test_detection/data"

def do_3d():
    # Load the 3-dimensional lower bound and the 3-dimensional upper bound into stack.
    lower_list = []
    upper_list = []
    scales = tuple([0.5 * 1.6 ** k for k in range(6)])
    nscales = len(scales)
    for i in range(nscales):
        lower = np.loadtxt(LOC + f"/lower{i}.csv", delimiter=",")
        upper = np.loadtxt(LOC + f"/upper{i}.csv", delimiter=",")
        lower_list.append(lower)
        upper_list.append(upper)
    lower3d = np.array(lower_list)
    upper3d = np.array(upper_list)
    nscales, m, n = lower3d.shape
    # Load scale-normalized 3-dimensional Laplacian
    print("Initializing normalized Laplacian")
    scale_laplacian = NormalizedLaplacian(m=m, n=n, scales=scales).mat
    print("Done.")
    minimal_bumps = minimize_bumps(lb=lower3d, ub=upper3d, g=scale_laplacian)
    # Plot 2d-slices of minimal_bumps on same scale.
    vmax = minimal_bumps.max()
    for i in range(nscales):
        fig = plt.figure()
        plt.imshow(minimal_bumps[i], cmap="gnuplot", vmax=vmax)
        fig.suptitle(f"h = {scales[i]}")
    plt.show()
    # Determine scale space maxima of Laplacian of minimal-bump-function.
    laplacian_minimal_bump_flat = scale_laplacian @ minimal_bumps.flatten()
    laplacian_minimal_bump = np.reshape(laplacian_minimal_bump_flat, (nscales, m, n))
    # Visualize the Laplacian
    for i in range(nscales):
        fig = plt.figure()
        plt.imshow(laplacian_minimal_bump[i], cmap="gnuplot", vmax=vmax)
        fig.suptitle(f"h = {scales[i]}")
    plt.show()
    # Determine scale space minima of laplacian
    local_minima = morphology.local_minima(image=laplacian_minimal_bump, indices=True, allow_borders=True)
    local_minima = np.array(local_minima).T
    # Visualize the local maxima in the MAP image.
    map_image = np.loadtxt(LOC + "/map.csv", delimiter=",")
    fig = plt.figure()
    blob_list = [blob for blob in local_minima]
    plotty_blobby(image=map_image, blobs=blob_list, scales=scales)
    plt.show()

print(os.getcwd())
do_3d()
