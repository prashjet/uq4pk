



import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode import ExponentialFilterFunction


def test_exponential_filter_function():
    m = 10
    n = 7
    # make filter function
    filter_function = ExponentialFilterFunction(m=m, n=n, a=1, b=1, c=2, d=2, h=2)
    for i in range(m * n):
        array_flat = np.zeros(m * n)
        # get the right filter
        filter = filter_function.filter(i)
        array_flat[filter.indices] = filter.weights
        filter_sum = np.sum(filter.weights)
        assert np.isclose(filter_sum, 1.)
        array = np.reshape(array_flat, (m, n))
        ax = plt.gca()
        ax.set_yticks(np.arange(-0.5, m, 1))
        ax.set_xticks(np.arange(-0.5, n, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='w', linestyle='-', linewidth=1)
        plt.imshow(array)
        plt.savefig(f"out/squared_exponential_{i}", bbox_inches="tight")
        print(f"{i+1}/{m * n}")

def test_downsampling():
    m = 6
    n = 10
    # make filter function
    filter_function = ExponentialFilterFunction(m=m, n=n, a=2, b=3, c=1, d=1)
    counter = 1
    for filter in filter_function.get_filter_list():
        array_flat = np.zeros(m * n)
        array_flat[filter.indices] = filter.weights
        print(f"LinearFilter sum = {np.sum(filter.weights)}")
        array = np.reshape(array_flat, (m, n))
        ax = plt.gca()
        ax.set_yticks(np.arange(-0.5, m, 1))
        ax.set_xticks(np.arange(-0.5, n, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='w', linestyle='-', linewidth=1)
        plt.imshow(array)
        plt.savefig(f"out/downsampling_{counter}", bbox_inches="tight")
        print(f"{counter}/{filter_function.size}")
        counter += 1

test_exponential_filter_function()