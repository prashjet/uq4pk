from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.blob_detection.extra import DOH
from uq4pk_fit.blob_detection.doh import doh_blanket
from uq4pk_fit.blob_detection import doh


def test_doh_op():
    m = 12
    n = 53
    doh_op = DOH(m=m, n=n)
    test_im = np.loadtxt("data/map.csv", delimiter=",")
    test_vec = test_im.flatten()
    doh_test = doh_op.fun(test_vec)
    doh_im = np.reshape(doh_test, (m, n))
    vmax = doh_im.max()
    vmin = doh_im.min()
    plt.figure(figsize=(6, 2.5))
    plt.imshow(doh_im, cmap="gnuplot", aspect="auto", vmax=vmax, vmin=vmin)
    plt.figure(figsize=(6, 2.5))
    doh_im2 = doh(test_im)
    plt.imshow(doh_im2, cmap="gnuplot", aspect="auto", vmax=vmax, vmin=vmin)
    plt.show()


def test_2d():
    # Load lower and upper bounds from files.
    lower = np.loadtxt("data/test_lower.csv", delimiter=",")
    upper = np.loadtxt("data/test_upper.csv", delimiter=",")
    blanket = doh_blanket(lower, upper)
    doh_of_blanket = doh(blanket)
    # Make pictures
    cmap = plt.get_cmap("gnuplot")
    vmin = 0.
    vmax = upper.max()
    fignum = 0
    names = ["Lower bound", "Upper bound", "Blanket",]
    for im, name in zip([lower, upper, blanket], names):
        fig = plt.figure(num=fignum, figsize=(6, 2.5))
        ax = plt.axes()
        ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.suptitle(name)
        fignum += 1
    fig = plt.figure(figsize=(6, 2.5))
    plt.imshow(doh_of_blanket, label="DOH of blanket")
    plt.show()