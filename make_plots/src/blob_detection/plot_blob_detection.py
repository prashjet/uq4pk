
from matplotlib import pyplot as plt
plt.style.use("src/uq4pk.mplstyle")
import numpy as np
from pathlib import Path

import uq4pk_src
from uq4pk_fit.visualization import plot_distribution_function, plot_significant_blobs, plot_blobs
from uq4pk_fit.blob_detection import laplacian_of_gaussians, ulog, compute_blanket
from ..plot_params import CW
from .parameters import SIGMA_LIST, MAP, LOWER_STACK, UPPER_STACK, RTHRESH1, RTHRESH2,\
    OVERLAP1, OVERLAP2, BLANKET_SCALE
from .one_dimensional_example import one_dimensional_example

from ..util import add_colorbar_to_axis


# Define plot names
name_log_demo = "log_demo.png"
name_blanket = "blanket.png"
name_one_dimensional = "one_dimensional_example.png"
name_ulog_demo = "ulog_demo.png"
name_speedup_comparison = "comparison_exact_vs_speedup.png"

ssps = uq4pk_src.model_grids.MilesSSP()


def plot_blob_detection(src: Path, out: Path):
    _log_demo(src, out)             # Creates figure 2
    _blanket(src, out)
    _one_dimensional(src, out)      # Creates figure 5
    _ulog_demo(src, out)            # Creates figure 6


def _log_demo(src: Path, out: Path):
    """
    Creates figure 2 for the paper.
    """
    # Get MAP.
    f_map = np.load(str(src / MAP))
    # Apply deterministic LoG.
    map_blobs = laplacian_of_gaussians(image=f_map, sigma_list=SIGMA_LIST, rthresh=RTHRESH1, max_overlap=OVERLAP1,
                                       exclude_max_scale=False)
    # Create plot.
    fig = plt.figure(figsize=(CW, 0.8 * CW))
    ax = plt.gca()
    im = plot_blobs(ax=ax, image=f_map, blobs=map_blobs, ssps=ssps, flip=False)
    # Add colorbar to image.
    add_colorbar_to_axis(fig, ax, im)

    plt.savefig(str(out / name_log_demo), bbox_inches="tight")


def _blanket(src: Path, out: Path):
    """
    Creates figure 5 of the paper (two-dimensional image of a t-blanket).
    """
    lower_stack_path = LOWER_STACK
    upper_stack_path = UPPER_STACK
    # Load precomputed arrays
    lower_stack = np.load(str(src / lower_stack_path))
    upper_stack = np.load(str(src / upper_stack_path))
    # Get lower and upper FCI at desired scale.
    fci_low = lower_stack[BLANKET_SCALE]
    fci_upp = upper_stack[BLANKET_SCALE]
    # Compute blanket.
    blanket = compute_blanket(lb=fci_low, ub=fci_upp)
    # Plot blanket.
    fig = plt.figure(figsize=(CW, 0.8 * CW))
    ax = plt.axes()
    im = plot_distribution_function(ax=ax, image=blanket, ssps=ssps, flip=False, xlabel=True, ylabel=True)
    add_colorbar_to_axis(fig, ax, im)
    # Store image.
    plt.savefig(str(out / name_blanket), bbox_inches="tight")


def _one_dimensional(src: Path, out: Path):
    """
    Creates figure 6 for the paper.
    """
    lower1d, upper1d, map1d, second_order_string = one_dimensional_example()
    # Visualize result.
    n = lower1d.size
    fig = plt.figure(figsize=(CW, 0.5 * CW))
    ax = plt.axes()
    x_span = np.arange(start=0, stop=1, step=(1 / n))
    # First, plot without strings.
    plt.rc('text', usetex=True)
    ax.plot(x_span, lower1d, color="b", linestyle="--", label=r"$L_t^\mathrm{low}$")
    ax.plot(x_span, upper1d, color="b", linestyle="-.", label=r"$L_t^\mathrm{upp}$")
    plt.plot(x_span, map1d, label=r"$L^\mathrm{MAP}_t$", linestyle=":", color="r")
    ax.plot(x_span, second_order_string, label=r"$\bar B_t$", color="g")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    # Shade area between lower1d and upper1d
    ax.fill_between(x_span, lower1d, upper1d, alpha=0.2)
    # Remove ticks
    plt.yticks([], [])

    plt.savefig(str(out / name_one_dimensional), bbox_inches="tight")


def _ulog_demo(src: Path, out: Path):
    """
    Creates figure 3 for the paper.
    """
    fig = plt.figure(figsize=(CW, 0.8 * CW))
    ax = plt.axes()
    # Plot blob detection (without speedup, with colorbar).
    im = _plot_blobs_from_stack(ax, src=src)
    # Add colorbar.
    add_colorbar_to_axis(fig, ax, im)

    plt.savefig(str(out / name_ulog_demo), bbox_inches="tight")


def _plot_blobs_from_stack(ax, src: Path, xlabel=True, ylabel=True):

    lower_stack_path = LOWER_STACK
    upper_stack_path = UPPER_STACK

    # Load precomputed arrays
    f_map = np.load(str(src / MAP))
    lower_stack = np.load(str(src / lower_stack_path))
    upper_stack = np.load(str(src / upper_stack_path))

    # Perform uncertainty-aware blob detection.
    significant_blobs = ulog(sigma_list=SIGMA_LIST, lower_stack=lower_stack, upper_stack=upper_stack, reference=f_map,
                             rthresh1=RTHRESH1, rthresh2=RTHRESH2, overlap1=OVERLAP1, overlap2=OVERLAP2,
                             exclude_max_scale=False)

    # Create plot.
    im = plot_significant_blobs(ax=ax, image=f_map, blobs=significant_blobs, ssps=ssps, flip=False, xlabel=xlabel,
                                ylabel=ylabel)
    return im

