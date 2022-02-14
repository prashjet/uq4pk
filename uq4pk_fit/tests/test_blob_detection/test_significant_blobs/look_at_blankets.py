
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.blob_detection.scale_normalized_laplacian import scale_normalized_laplacian


N = 12
SIGMA_MAX = 15
SIGMA_MIN = 1
i = 6


def show_blanket_slices():

    lower_slices = []
    upper_slices = []
    blanket_slices = []

    for k in range(N):
        lower_k = np.loadtxt(f"data/lower{k}.csv", delimiter=",")
        upper_k = np.loadtxt(f"data/upper{k}.csv", delimiter=",")
        blanket_k = np.loadtxt(f"data/blanket{k}.csv", delimiter=",")

        # make Slices.
        n = lower_k.shape[1]
        lower_k_slice = lower_k[i].reshape((n, ))
        upper_k_slice = upper_k[i].reshape((n, ))
        blanket_k_slice = blanket_k[i].reshape((n, ))

        # Append to lists.
        lower_slices.append(lower_k_slice)
        upper_slices.append(upper_k_slice)
        blanket_slices.append(blanket_k_slice)

    # Plot all slices on same scale.
    n = lower_slices[0].size
    for k in range(N):

        # plot slices
        plt.figure(num=k)
        x_axis = np.arange(n)
        names = ["Lower bound", "Upper bound", "Laplacian blanket"]
        colors = ["red", "orange", "lime"]
        for slice, name, color in zip([lower_slices[k], upper_slices[k], blanket_slices[k]],
                                      names, colors):
            plt.plot(x_axis, slice, label=name, color=color)
        plt.legend()

    plt.show()


def show_blanket_Laplacians():

    sigma_step = (SIGMA_MAX - SIGMA_MIN) / (N - 1)
    sigmas = [SIGMA_MIN + n * sigma_step for n in range(N)]
    scales = [0.5 * s ** 2 for s in sigmas]

    # Compute the scale normalized-Laplacian of the blanket stack.

    blanket_list = []
    for k in range(N):
        blanket = np.loadtxt(f"data/blanket{k}.csv", delimiter=",")
        blanket_list.append(blanket)
    blanket_stack = np.array(blanket_list)

    # Compute scale-normalized Laplacian
    snl = scale_normalized_laplacian(blanket_stack, scales, mode="reflect")
    # Invert for better intuition
    snl = - snl
    # Show every other slice
    n = snl.shape[2]
    snl_max = snl.max()
    snl_min = snl.min()
    h_span = np.arange(n)
    for j in range(int(N / 2)):
        k = 2 * j + 1
        slice_k = snl[k][i].reshape((n, ))
        plt.figure(num=k)
        plt.ylim(snl_min, snl_max)
        plt.plot(h_span, slice_k, label="Laplacian blanket")
        plt.legend()
    plt.show()


show_blanket_Laplacians()