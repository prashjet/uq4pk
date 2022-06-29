
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.visualization import plot_distribution_function


def test_plot_distribution_function():
    test_img = np.loadtxt("map.csv", delimiter=",")
    plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    plot_distribution_function(image=test_img, ax=ax)
    plt.show()