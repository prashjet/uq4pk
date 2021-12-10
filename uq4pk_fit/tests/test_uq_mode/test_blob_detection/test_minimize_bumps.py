
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteLaplacian
from uq4pk_fit.uq_mode.blob_detection.minimize_bumps import minimize_bumps
from uq4pk_fit.uq_mode.blob_detection.cheap_blanket import cheap_blanket


def test_1d():
    # Get lb and ub as slices from lower and upper bound
    # Load lower and upper bounds from files.
    lower = np.loadtxt("data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("data/test_upper.csv", delimiter=",")
    i = [8]
    n = lower.shape[1]
    lower_slice = lower[i, :].reshape((n,))
    upper_slice = upper[i, :].reshape((n,))
    # Load Laplacian
    lap = DiscreteLaplacian(shape=(n, )).mat
    minimal_bumps = minimize_bumps(lb=lower_slice, ub=upper_slice, g=lap)
    snd_deriv_flat = lap @ minimal_bumps.flatten()
    snd_deriv = np.reshape(snd_deriv_flat, lower_slice.shape)
    # For comparison, also compute blanket
    blanket = cheap_blanket(lb=lower_slice, ub=upper_slice)
    blanket_dxx = lap @ blanket
    # Visualize result.
    fig = plt.figure()
    x_span = np.arange(n)
    #print(f"Approximation error = {np.linalg.norm(string - cheap_string)}")
    names = ["upper bound", "lower bound", "minimal bumps", "blanket"]
    for slice, name in zip([lower_slice, upper_slice, minimal_bumps, blanket], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    # Also compare second derivatives.
    names = ["Second derivative of minimal bumps", "Second derivative of blanket"]
    fig = plt.figure()
    for slice, name in zip([snd_deriv, blanket_dxx], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    plt.show()


def test_2d():
    # Load lower and upper bounds from files.
    lower = np.loadtxt("data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("data/test_upper.csv", delimiter=",")
    # Load Laplacian
    lap = DiscreteLaplacian(shape=lower.shape).mat
    minimal_bumps = minimize_bumps(lb=lower, ub=upper, g=lap)
    blanket = cheap_blanket(lower, upper)
    # Make pictures
    cmap = plt.get_cmap("gnuplot")
    vmin = 0.
    vmax = upper.max()
    fignum = 0
    names = ["Lower bound", "Upper bound", "Minimal bumps", "Blanket"]
    for im, name in zip([lower, upper, minimal_bumps, blanket], names):
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
    minbump_slice = minimal_bumps[i, :].reshape((n,))
    blanket_slice = blanket[i, :].reshape((n,))
    x_span = np.arange(lower_slice.size)
    fig = plt.figure()
    names = ["upper bound", "lower bound", "Minimal bumps", "Blanket"]
    for slice, name in zip([lower_slice, upper_slice, minbump_slice, blanket_slice], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    plt.show()
