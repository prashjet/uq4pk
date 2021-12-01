
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode.detection.span_blanket import span_blanket
from uq4pk_fit.uq_mode.detection.cheap_blanket import cheap_blanket


def test_1d():
    # Get lb and ub as slices from lower and upper bound
    # Load lower and upper bounds from files.
    lower = np.loadtxt("data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("data/test_upper.csv", delimiter=",")
    i = [8]
    n = lower.shape[1]
    lower_slice = lower[i, :].reshape((n,))
    upper_slice = upper[i, :].reshape((n,))
    # Compute taut string
    string = span_blanket(lower_slice, upper_slice)
    # Compute string cheaply
    cheap_string = cheap_blanket(lower_slice, upper_slice)
    # Visualize result.
    fig = plt.figure(num=0)
    x_span = np.arange(n)
    print(f"Approximation error = {np.linalg.norm(string - cheap_string)}")
    names = ["upper bound", "lower bound", "exact computation", "approximate computation"]
    for slice, name in zip([lower_slice, upper_slice, string, cheap_string], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    plt.show()


def test_2d():
    # Load lower and upper bounds from files.
    lower = np.loadtxt("data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("data/test_upper.csv", delimiter=",")
    # Compute taut string.
    taut = span_blanket(lower, upper)
    cheap_taut = cheap_blanket(lower, upper)
    # Make pictures
    cmap = plt.get_cmap("gnuplot")
    vmin = 0.
    vmax = upper.max()
    fignum = 0
    names = ["Lower bound", "Upper bound", "Blanket", "Approximate Blanket"]
    for im, name in zip([lower, upper, taut, cheap_taut], names):
        fig = plt.figure(num=fignum, figsize=(6, 2.5))
        ax = plt.axes()
        ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.suptitle(name)
        fignum += 1
    print(f"Approximation error = {np.linalg.norm(taut - cheap_taut)}")
    # Also plot y-slice
    i = [6]
    n = lower.shape[1]
    lower_slice = lower[i, :].reshape((n,))
    upper_slice = upper[i, :].reshape((n,))
    taut_slice = taut[i, :].reshape((n,))
    cheap_slice = cheap_taut[i, :].reshape((n, ))
    x_span = np.arange(lower_slice.size)
    fig = plt.figure()
    names = ["upper bound", "lower bound", "exact computation", "approximate computation"]
    for slice, name in zip([lower_slice, upper_slice, taut_slice, cheap_slice], names):
        plt.plot(x_span, slice, label=name)
    plt.legend()
    plt.show()