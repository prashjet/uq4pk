import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np


def plot_triple_bar(safename, name_list, values1, values2, values3, name1, name2, name3, errorbars=None):
    """
    Creates a double-bar chart with error bars.
    """
    bar_width = 0.2
    grid = np.arange(len(name_list))
    fig = plt.figure()
    # Plot reference values
    plt.bar(grid - bar_width, values1, bar_width, label=name1)
    # Plot MAP (optionally with errorbars)
    if errorbars is None:
        plt.bar(grid, values2, bar_width, label=name2)
    else:
        plt.bar(grid, values2, bar_width, label=name2, yerr=errorbars, capsize=5)
    # Plot ground truth
    plt.bar(grid + bar_width, values3, bar_width, label=name3)
    plt.xticks(grid, name_list)
    plt.xlabel("Parameter")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{safename}.png", bbox_inches="tight")
    plt.close()