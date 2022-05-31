
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.gaussian_blob.scale_space_representation.scale_space_representation import scale_space_representation
from uq4pk_fit.gaussian_blob.scale_space_representation.scale_normalized_laplacian import scale_normalized_laplacian


SHOW = True    # Set True if you want to see plots
NSCALES = 12
R_MIN = 1
R_MAX = 15


def test_normalized_laplacian():
    # Load test image
    testim = np.loadtxt("data/test.csv", delimiter=",")

    # Create scale-space representation.
    r_step = (R_MAX - R_MIN) / NSCALES
    radii = [R_MIN + n * r_step for n in range(NSCALES)]
    scales = [0.5 * r ** 2 for r in radii]
    # Compute scale space representation
    ssr = scale_space_representation(testim, scales=scales, mode="constant")

    # Apply scale-normalized Laplacian
    snl = scale_normalized_laplacian(ssr, scales, mode="reflect")

    # Visualize
    i = 0
    vmin = snl.min()
    vmax = snl.max()
    for snl_i, ssr_i in zip(snl, ssr):
        fig = plt.figure(num=f"gaussian, t = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(ssr_i, cmap="gnuplot", vmin=vmin, vmax=vmax)
        fig = plt.figure(num=f"t = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(snl_i, cmap="gnuplot", vmin=vmin, vmax=vmax)
        i += 1
    if SHOW: plt.show()


def test_blankets():
    # Load blanket stack
    r_step = (R_MAX - R_MIN) / (NSCALES - 1)
    sigmas = [1 + n * r_step for n in range(NSCALES)]
    scales = [0.5 * r ** 2 for r in sigmas]
    blanket_list = []
    for i in range(NSCALES):
        blanket = np.loadtxt(f"data/blanket{i}.csv", delimiter=",")
        blanket_list.append(blanket)
    blanket_stack = np.array(blanket_list)

    # Compute scale-normalized Laplacian
    snl = scale_normalized_laplacian(blanket_stack, scales, mode="reflect")

    # Visualize
    i = 0
    vmin = snl.min()
    vmax = snl.max()
    for snl_i in snl:
        fig = plt.figure(num=f"sigma = {sigmas[i]}", figsize=(6, 2.5))
        plt.imshow(snl_i, cmap="gnuplot", vmin=vmin, vmax=vmax)
        i += 1
    if SHOW: plt.show()


def test_with_ratio():
    # Load test image
    testim = np.loadtxt("data/test.csv", delimiter=",")

    # Create scale-space representation.
    r_step = (R_MAX - R_MIN) / NSCALES
    sigmas = [R_MIN + n * r_step for n in range(NSCALES)]
    scales = [0.5 * r ** 2 for r in sigmas]
    # Compute scale space representation
    ssr = scale_space_representation(testim, scales=scales, mode="reflect")

    # Apply scale-normalized Laplacian
    snl = scale_normalized_laplacian(ssr, scales, mode="reflect")

    # Visualize
    i = 0
    vmin = snl.min()
    vmax = snl.max()
    for snl_i in snl:
        fig = plt.figure(num=f"t = {scales[i]}", figsize=(6, 2.5))
        plt.imshow(snl_i, cmap="gnuplot", vmin=vmin, vmax=vmax)
        i += 1
    if SHOW: plt.show()