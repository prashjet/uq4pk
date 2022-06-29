"""
Creates figures for the "computational_aspect" section.
"""


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .grids import plot_localization_window, plot_twolevel_discretization
from .parameters import ERRORS_WINDOW_FILE, ERRORS_TWOLEVEL_FILE, HEURISTIC_WINDOW_FILE1, \
    HEURISTIC_WINDOW_FILE2, N1, N2


# Names of plots.
grids = "grids.png"
speedup_errors = "speedup_errors.png"


def plot_computational_aspects(src: Path, out: Path):
    #_plot_windows(src, out)
    _plot_error_graphs(src, out)


def _plot_windows(src: Path, out: Path):
    """
    Creates figure 8 in the paper.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Plot a pixel with its localization window.
    plot_localization_window(ax[0], w1=4, w2=4)
    plot_twolevel_discretization(ax[1], w1=1, w2=1, d1=3, d2=3)

    plt.savefig(str(out / grids), bbox_inches="tight")
    plt.show()


def _plot_error_graphs(src: Path, out: Path):
    """
    Creates figure 9.
    """
    num_cpus = 8
    ylim = [1e-3, 1.]
    xlim = [0, 150]

    results_window = np.load(str(src / ERRORS_WINDOW_FILE))
    times_window = results_window[0] / num_cpus
    errors_window = results_window[1]
    # Load results for heuristic with small sample size.
    results_window_heuristic1 = np.load(str(src / HEURISTIC_WINDOW_FILE1))
    times_window_heuristic1 = results_window_heuristic1[0] / num_cpus
    errors_window_heuristic1 = results_window_heuristic1[1]
    # Load results for heuristic with moderate sample size..
    results_window_heuristic2 = np.load(str(src / HEURISTIC_WINDOW_FILE2))
    times_window_heuristic2 = results_window_heuristic2[0] / num_cpus
    errors_window_heuristic2 = results_window_heuristic2[1]

    # START PLOTTING:

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot localization vs twolevel.
    results_twolevel = np.load(src / ERRORS_TWOLEVEL_FILE)
    times_twolevel = results_twolevel[0] / num_cpus
    errors_twolevel = results_twolevel[1]
    # y axis should be plotted logarithmically
    ax[0].set_yscale("log")
    # Because we are plotting logarithmically, we remove the last entries that correspond to 0 approximation error.
    ax[0].plot(times_window[:-1], errors_window[:-1], label="localization window", marker="o",
               linestyle="-")
    ax[0].plot(times_twolevel[:-1], errors_twolevel[:-1],
               label="non-uniform grid", marker="o", linestyle="-")
    # The time for the exact computations is represented using a vertical line.
    ax[0].axvline(x=times_twolevel[-1], linestyle=":")
    ax[0].set_xlabel("computation time [s]")
    ax[0].set_ylabel("mean Jaccard distance")
    ax[0].set_yticks([1e-1, 5e-2, 1e-2])
    ax[0].set_yticklabels(["$1 \cdot 10^{-1}$", "$5 \cdot 10^{-2}$", "$1 \cdot 10^{-2}$"])
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].legend()

    # Plot true vs heuristic.
    ax[1].set_yscale("log")
    ax[1].plot(times_window[:-1], errors_window[:-1], label="localization error", marker="o",
             linestyle="-")
    ax[1].plot(times_window_heuristic1[:-1], errors_window_heuristic1[:-1],
             label="heuristic, $N_\mathrm{sample} = $" + str(N1),
             marker="v", linestyle="--")
    ax[1].plot(times_window_heuristic2[:-1], errors_window_heuristic2[:-1],
             label="heuristic, $N_\mathrm{sample} = $" + str(N2),
             marker="v", linestyle="--")
    ax[1].set_xlabel("computation time [s]")
    # Reduce labels for y-axis in second plot.
    ax[1].set_yticks([1e-1, 5e-2, 1e-2])
    ax[1].set_yticklabels(["$1 \cdot 10^{-1}$", "$5 \cdot 10^{-2}$", "$1 \cdot 10^{-2}$"])
    ax[1].axvline(x=times_twolevel[-1], linestyle=":")
    ax[1].set_ylim(ylim)
    ax[1].set_xlim(xlim)
    ax[1].legend()
    # Save and show
    plt.savefig(str(out / speedup_errors), bbox_inches="tight")
    plt.show()







