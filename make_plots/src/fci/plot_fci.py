import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from uq4pk_fit.visualization import plot_distribution_function
import uq4pk_src
from ..plot_params import CW2
from .parameters import MAPFILE, GROUND_TRUTH, PCILOW, PCIUPP, FCILOW, FCIUPP, FILTERED_MAP, DV, SCALES
from ..util import add_colorbar_to_plot, add_colorbar_to_axis


# Use consistent style.
plt.style.use("src/uq4pk.mplstyle")

# Names for the plot.
plot_name = "credible_intervals.png"

# Set up ssps-grid and other requirements.
ssps = uq4pk_src.model_grids.MilesSSP()
ssps.logarithmically_resample(dv=DV)


def plot_fci(src: Path, out: Path):

    # -- Load data.
    f_map = np.load(str(src / MAPFILE) + ".npy")
    f_true = np.load(str(src / GROUND_TRUTH) + ".npy")
    pci_low = np.load(str(src / PCILOW) + ".npy")
    pci_upp = np.load(str(src / PCIUPP) + ".npy")
    # Load FCIs and filtered MAPs.
    map_list = []
    upper_list = []
    lower_list = []
    for lowname, mapname, uppname in zip(FCILOW, FILTERED_MAP, FCIUPP):
        lower_list.append(np.load(str(src / lowname) + ".npy"))
        map_list.append(np.load(str(src / mapname) + ".npy"))
        upper_list.append(np.load(str(src / uppname) + ".npy"))

    # -- Create figure object.
    fig = plt.figure(figsize=(CW2, CW2))

    # -- Plot ground truth in first row.
    cb_axes = []
    mappables = []
    vmax = pci_upp.max()
    ax1 = plt.subplot2grid(shape=(5, 3), loc=(0, 0), colspan=3)
    ax1.set_title("Ground truth")
    im1 = plot_distribution_function(ax1, image=f_true, ssps=ssps, xlabel=True, ylabel=True, vmax=vmax, flip=False)
    mappables.append(im1)
    cb_axes.append(ax1)

    # -- Second row: PCIs and MAP.
    ax_left = plt.subplot2grid(shape=(5, 3), loc=(1, 0), colspan=1)
    ax_left.set_title(r"$f^\mathrm{low}$")
    plot_distribution_function(ax_left, image=pci_low, ssps=ssps, xlabel=False, ylabel=True, vmax=vmax, flip=False)
    ax_middle = plt.subplot2grid(shape=(5, 3), loc=(1, 1), colspan=1)
    ax_middle.set_title(r"$f^\mathrm{MAP}$")
    plot_distribution_function(ax_middle, image=f_map, ssps=ssps, xlabel=False, ylabel=False, vmax=vmax, flip=False)
    ax_right = plt.subplot2grid(shape=(5, 3), loc=(1, 2), colspan=1)
    ax_right.set_title(r"$f^\mathrm{upp}$")
    im_right = plot_distribution_function(ax_right, image=pci_upp, ssps=ssps, xlabel=False, ylabel=False, vmax=vmax,
                                          flip=False)
    mappables.append(im_right)
    cb_axes.append(ax_right)


    # -- Following rows: Filtered MAP and FCIs.
    i = 0
    for lower, map, upper in zip(lower_list, map_list, upper_list):
        if i == 2:
            xlabel = True
        else:
            xlabel = False
        vmax = upper.max()
        t = SCALES[i]
        ax_left = plt.subplot2grid(shape=(5, 3), loc=(i + 2, 0), colspan=1)
        plot_distribution_function(ax_left, image=lower, ssps=ssps, xlabel=xlabel, ylabel=True, vmax=vmax, flip=False)
        ax_middle = plt.subplot2grid(shape=(5, 3), loc=(i + 2, 1), colspan=1)
        plot_distribution_function(ax_middle, image=map, ssps=ssps, xlabel=xlabel, ylabel=False, vmax=vmax, flip=False)
        ax_right = plt.subplot2grid(shape=(5, 3), loc=(i + 2, 2), colspan=1)
        im_right = plot_distribution_function(ax_right, image=upper, ssps=ssps, xlabel=xlabel, ylabel=False, vmax=vmax,
                                              flip=False)
        mappables.append(im_right)
        cb_axes.append(ax_right)
        ax_left.set_title(r'$L_{{{:.0f}}}^\mathrm{{{}}}$'.format(t, "low"))
        ax_middle.set_title(r'$L_{{{:.0f}}}^\mathrm{{{}}}$'.format(t, "MAP"))
        ax_right.set_title(r'$L_{{{:.0f}}}^\mathrm{{{}}}$'.format(t, "upp"))
        i += 1

    # We need extra vertical space, because otherwise the axis labels overlap.
    plt.tight_layout()
    # Only now can we add the colorbars.
    for i in range(5):
        add_colorbar_to_axis(fig, cb_axes[i], mappables[i])

    # Save the plot.
    plt.savefig(str(out / plot_name), bbox_inches="tight")