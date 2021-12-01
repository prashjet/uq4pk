import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
from uq4pk_src.observation_operator import ObservationOperator


def plot_with_colorbar(image, ticks=None, savename=None, vmax=None, vmin=None, show=False):
    """
    Plots the age-metallicity distribution function with a colorbar on the side that
    shows which color belongs to which value.
    :param image: ndarray
    :param savename: string
    :param vmax: float
    """
    cmap = plt.get_cmap("gnuplot")
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = plt.imshow(image, vmax=vmax, vmin=vmin, cmap=cmap, extent=(0, 1, 0, 1), aspect="auto")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("density")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Metallicity [Z/H]")
    if ticks is not None:
        t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
        ax.set_xticks(img_t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_yticks(img_z_ticks)
        ax.set_yticklabels(z_ticks)
    if show: plt.show()
    # if savename is provided, save .csv and image
    if savename is not None:
        np.savetxt(f"{savename}.csv", image, delimiter=',')
        plt.savefig(f"{savename}.png", bbox_inches='tight')
    plt.close()