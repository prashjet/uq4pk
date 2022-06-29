import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from uq4pk_fit.visualization import plot_distribution_function
import uq4pk_src
from .parameters import MAPFILE, GROUND_TRUTH, PCILOW, PCIUPP, FCILOW, FCIUPP, DV

from ..util import add_colorbar_to_plot, add_colorbar_to_axis


# Names for the plots
pci_plot = "pci.png"
fci_plot = "fcis.png"

# Set up ssps-grid and other requirements.
ssps = uq4pk_src.model_grids.MilesSSP()
ssps.logarithmically_resample(dv=DV)


def plot_fci(src: Path, out: Path):
    # Create figure 3.
    _plot_pci(src, out)
    # Create figure 4
    _plot_fcis(src, out)


def _plot_pci(src: Path, out: Path):
    """
    Creates figure 3 for the paper. Contrast ground truth and MAP reconstruction to pixelwise credible bands.
    """
    # Get the required arrays.
    f_map = np.load(str(src / MAPFILE) + ".npy")
    f_true = np.load(str(src / GROUND_TRUTH) + ".npy")
    pci_low = np.load(str(src / PCILOW) + ".npy")
    pci_upp = np.load(str(src / PCIUPP) + ".npy")
    # Maximum value is maximum of pci upper bound.
    vmax = pci_upp.max()
    # Start plotting.
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    plot_distribution_function(ax[0, 0], image=f_true, ssps=ssps, vmax=vmax, flip=False, xlabel=False, ylabel=True)
    plot_distribution_function(ax[1, 0], image=f_map, ssps=ssps, vmax=vmax, flip=False, xlabel=True, ylabel=True)
    plot_distribution_function(ax[0, 1], image=pci_upp, ssps=ssps, vmax=vmax, flip=False, xlabel=False, ylabel=False)
    im = plot_distribution_function(ax[1, 1], image=pci_low, ssps=ssps, vmax=vmax, flip=False, xlabel=True,
                                    ylabel=False)
    # Add colorbar.
    add_colorbar_to_plot(fig, ax, im)

    # Save and show.
    plt.savefig(str(out / pci_plot), bbox_inches="tight")
    plt.show()


def _plot_fcis(src: Path, out: Path):
    """
    Creates figure 4 for the paper. Shows filtered credible bands for 3 different scales.
    """
    # Get FCIs
    fci_low_list = []
    fci_upp_list = []
    for lowname, uppname in zip(FCILOW, FCIUPP):
        fci_low_list.append(np.load(str(src / lowname) + ".npy"))
        fci_upp_list.append(np.load(str(src / uppname) + ".npy"))
    assert len(fci_low_list) == len(fci_upp_list) == 3
    # Create plot.
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    i = 0
    for fci_low, fci_upp in zip(fci_low_list, fci_upp_list):
        vmax_t = fci_upp.max()
        if i == 2:
            xlabel = True
        else:
            xlabel = False
        # Plot lower bound.
        plot_distribution_function(ax[i, 0], image=fci_low, vmax=vmax_t, ssps=ssps, flip=False, xlabel=xlabel,
                                   ylabel=True)
        # Plot upper bound.
        im = plot_distribution_function(ax[i, 1], image=fci_upp, vmax=vmax_t, ssps=ssps, flip=False, xlabel=xlabel,
                                        ylabel=False)
        # Add colorbar to right image.
        add_colorbar_to_axis(fig, ax[i, 1,], im)
        i += 1

    # Save and show.
    plt.savefig(str(out / fci_plot), bbox_inches="tight")
    plt.show()