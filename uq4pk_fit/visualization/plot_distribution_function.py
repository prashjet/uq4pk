
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from .params import power_norm, CMAP


def plot_distribution_function(ax: Axes, image, ssps = None, vmax=None, flip=True, xlabel=True, ylabel=False):
    """
    Plots the age-metallicity distribution function with a colorbar on the side that
    shows which color belongs to which value.
    :param ax: A matplotlib.axes.Axes object.
    :param image: The age-metallicity distribution as 2-dimensional numpy array.
    :param ssps: The SSPS grid.
    :param vmax: The maximum intensity.
    :param flip: If True, the plotted image is upside down. This is True by default, since it is more correct from a
        physical point of view.
    :param xlabel: If True, adds label to x-axis.
    """
    if flip:
        f_im = np.flipud(image)
    else:
        f_im = image
    cmap = plt.get_cmap(CMAP)
    if vmax is None:
        vmax = image.max()
    # I want fixed aspect ratio to 6:2.5.
    aspect = 2.5 / 6.
    immap = ax.imshow(f_im, cmap=cmap, extent=(0, 1, 0, 1), aspect=aspect, norm=power_norm(vmin=0., vmax=vmax))
    if xlabel:
        ax.set_xlabel("Age [Gyr]")
    if ylabel:
        ax.set_ylabel("Metallicity [Z/H]")
    if ssps is not None:
        ticks = [ssps.t_ticks, ssps.z_ticks, ssps.img_t_ticks, ssps.img_z_ticks]
        t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
        ax.set_xticks(img_t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_yticks(img_z_ticks)
        ax.set_yticklabels(z_ticks)

    # Return "mappable" (allows colorbar creation).
    return immap