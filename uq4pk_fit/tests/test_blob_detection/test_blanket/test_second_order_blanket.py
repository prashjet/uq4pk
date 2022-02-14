
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteLaplacian
from uq4pk_fit.blob_detection.blankets import second_order_blanket


def test_1d():
    # Get lb and ub as slices from lower and upper bound
    # Load lower and upper bounds from files.
    lower = np.loadtxt("../data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("../data/test_upper.csv", delimiter=",")
    i = [8]
    n = lower.shape[1]
    lower_slice = lower[i, :].reshape((n,))
    upper_slice = upper[i, :].reshape((n,))
    # Load Laplacian
    lap = DiscreteLaplacian(shape=(n, )).mat
    blanket = second_order_blanket(lb=lower_slice, ub=upper_slice)
    snd_deriv_flat = lap @ blanket.flatten()
    snd_deriv = np.reshape(snd_deriv_flat, lower_slice.shape)
    # Visualize result.
    fig = plt.figure()
    x_span = np.arange(n)
    #print(f"Approximation error = {np.linalg.norm(string - cheap_string)}")
    names = ["upper bound", "lower bound", "second-order blanket"]
    for slice, name in zip([lower_slice, upper_slice, blanket], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    # Also compare second derivatives.
    names = ["Second derivative of minimal bumps"]
    fig = plt.figure()
    for slice, name in zip([snd_deriv], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    plt.show()


def test_2d():
    # Load lower and upper bounds from files.
    lower = np.loadtxt("../data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("../data/test_upper.csv", delimiter=",")
    blanket = second_order_blanket(lb=lower, ub=upper)
    # Make pictures
    cmap = plt.get_cmap("gnuplot")
    vmin = 0.
    vmax = upper.max()
    fignum = 0
    names = ["Lower bound", "Upper bound", "2nd-order blanket"]
    for im, name in zip([lower, upper, blanket], names):
        fig = plt.figure(num=fignum, figsize=(6, 2.5))
        ax = plt.axes()
        ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.suptitle(name)
        fignum += 1
    # Also plot y-slice
    i = [6]
    n = lower.shape[1]
    lower_slice = lower[i, :].reshape((n,))
    upper_slice = upper[i, :].reshape((n,))
    blanket_slice = blanket[i, :].reshape((n,))
    x_span = np.arange(lower_slice.size)
    fig = plt.figure()
    names = ["upper bound", "lower bound", "blanket"]
    for slice, name in zip([lower_slice, upper_slice, blanket_slice], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    plt.show()
