
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteGradient
from uq4pk_fit.uq_mode.blob_detection.span_blanket import span_blanket


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
    # Visualize result.
    fig = plt.figure(num=0)
    x_span = np.arange(n)
    dg = DiscreteGradient([n]).mat
    def arclength(x):
        grad = dg @ x
        abs_grad = np.abs(grad)
        length = np.linalg.norm(np.sqrt(1 + np.square(abs_grad)), ord=1)
        return length
    print(f"Length: {arclength(string)}")
    for slice in [lower_slice, upper_slice, string]:
        plt.plot(x_span, slice)
    plt.show()


def test_2d():
    # Load lower and upper bounds from files.
    lower = np.loadtxt("data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("data/test_upper.csv", delimiter=",")
    # Compute taut string.
    taut = span_blanket(lower, upper)
    # Make pictures
    cmap = plt.get_cmap("gnuplot")
    vmin = 0.
    vmax = upper.max()
    fignum = 0
    for im in [lower, upper, taut]:
        fig = plt.figure(num=fignum, figsize=(6, 2.5))
        ax = plt.axes()
        ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
        fignum += 1
    # Also plot y-slice
    i = [6]
    n = lower.shape[1]
    lower_slice = lower[i, :].reshape((n,))
    upper_slice = upper[i, :].reshape((n,))
    taut_slice = taut[i, :].reshape((n,))
    # compare with 1 dimensional taut string
    string = span_blanket(lower_slice, upper_slice)
    x_span = np.arange(lower_slice.size)
    fig = plt.figure(num=3)
    for slice in [lower_slice, upper_slice, taut_slice, string]:
        plt.plot(x_span, slice)
    plt.show()

