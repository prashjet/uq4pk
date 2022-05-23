
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from typing import Sequence, Tuple, Union

from uq4pk_fit.gaussian_blob import GaussianBlob
from .params import CMAP, NORM


def plot_scaled_ellipse(ax, imshape, center, width, height, color, linestyle="-"):
    # Plot ellipse using normalized coordinates (a pixel is located at its center).
    vsize, hsize = imshape
    ncenter = ((center[0] + 0.5) / hsize, (vsize - center[1] - 0.5) / vsize)
    nwidth = width / hsize
    nheight = height / vsize
    ax.add_patch(patches.Ellipse(ncenter, width=nwidth, height=nheight, color=color, fill=False, linestyle=linestyle))


def plot_significant_blobs(image: np.ndarray, blobs: Sequence[Tuple[GaussianBlob, Union[GaussianBlob, None]]],
                           savefile: str = None, vmax: float = None, ssps = None, show: bool = False):
    """
    Makes a blob-plot for given image an blobs.

    :param image: The image as (m, dim)-array.
    :param blobs: A sequence of tuples. The first element corresponds to a blob detected in the image, while the
        second element is either None (the blob is not significant) or another blob, representing the significant feature.
    :param savefile: The name under which the plot should be saved (including file-type).
    :param vmin: Minimum intensity shown in plot.
    :param vmax: Maximum intensity shown in plot.
    :param ssps: The grid object needed for plotting.
    :param show: If False, not plot is shown.
    """
    if vmax is None:
        vmax = image.max()
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = plt.imshow(image, cmap=CMAP, aspect="auto", vmin=0., vmax=vmax, norm=NORM, extent=(0, 1, 0, 1))
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("density")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Metallicity [Z/H]")
    insignificant_color = "red"  # color for insignificant features
    feature_color = "lime"  # color for significant features -- inner part
    significant_color = "yellow"  # color for significant features -- outer part.

    for blob in blobs:
        b = blob[0]
        c = blob[1]
        if c is None:
            y, x = b.position
            plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x, y), width=b.width, height=b.height,
                                color=insignificant_color)
        else:
            # feature is significant
            y1, x1 = b.position
            y2, x2 = c.position
            # If the width and height of the significant blob agree with map blob, we increase the former slightly for
            # better visualization.
            factor = 1.05
            plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x1, y1), width=b.width, height=b.height,
                                color=feature_color)
            plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x2, y2), width=factor * c.width,
                                height=factor * c.height, color=significant_color, linestyle="--")
    # Currently, ticks don't work with patches.
    if ssps is not None:
        ticks = [ssps.t_ticks, ssps.z_ticks, ssps.img_t_ticks, ssps.img_z_ticks]
        t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
        ax.set_xticks(img_t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_yticks(img_z_ticks)
        ax.set_yticklabels(z_ticks)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_blobs(image: np.ndarray, blobs: Sequence[GaussianBlob], savefile: str = None, vmax: float = None, ssps = None,
               show: bool = False):
    """
    Makes a blob-plot for given image an blobs.

    :param image: The image as (m, dim)-array.
    :param blobs: A sequence of blobs.
    :param savefile: The name under which the plot should be saved (including file-type).
    :param vmax: Maximum intensity shown in plot.
    :param ssps: The grid object needed for plotting.
    :param show: If False, not plot is shown.
    """
    if vmax is None:
        vmax = image.max()
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = plt.imshow(image, cmap=CMAP, aspect="auto", vmin=0., vmax=vmax, norm=NORM, extent=(0, 1, 0, 1))
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("density")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Metallicity [Z/H]")
    blob_color = "lime"
    for blob in blobs:
        y, x = blob.position
        plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x, y), width=blob.width, height=blob.height,
                            color=blob_color)
    # Currently, ticks don't work with patches.
    if ssps is not None:
        ticks = [ssps.t_ticks, ssps.z_ticks, ssps.img_t_ticks, ssps.img_z_ticks]
        t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
        ax.set_xticks(img_t_ticks)
        ax.set_xticklabels(t_ticks)
        ax.set_yticks(img_z_ticks)
        ax.set_yticklabels(z_ticks)
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

