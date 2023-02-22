
from matplotlib.axes import Axes
from matplotlib import patches
import numpy as np
from typing import Sequence, Tuple, Union

from uq4pk_src.model_grids import MilesSSP
from uq4pk_fit.gaussian_blob import GaussianBlob
from .plot_distribution_function import plot_distribution_function
from .params import COLOR_BLOB, COLOR_INSIDE, COLOR_OUTSIDE, COLOR_SIGNIFICANT


def plot_scaled_ellipse(ax, imshape, center, width, height, color, linestyle="-", flip=True):
    """
    Plot ellipse using normalized coordinates (a pixel is located at its center).
    """
    vsize, hsize = imshape
    if flip:
        ncenter = ((center[0] + 0.5) / hsize, (center[1] + 0.5) / vsize)
    else:
        ncenter = ((center[0] + 0.5) / hsize, (vsize - center[1] - 0.5) / vsize)
    nwidth = width / hsize
    nheight = height / vsize
    ax.add_patch(patches.Ellipse(ncenter, width=nwidth, height=nheight, color=color, fill=False, linestyle=linestyle))


def plot_significant_blobs(ax: Axes, image: np.ndarray, blobs: Sequence[Tuple[GaussianBlob, Union[GaussianBlob, None]]],
                           vmax: float = None, ssps: MilesSSP = None, flip: bool = True, xlabel: bool = True,
                           ylabel: bool = True):
    """
    Makes a blob-plot for given image and blobs.

    Parameters
    ----------
    ax
        The axis object where the plot should be created.
    image : shape (m, dim)
        The reference image.
    blobs
        A sequence of tuples. The first element corresponds to a blob detected in the image, while the second element
        is either None (the blob is not significant) or another blob, representing the significant feature.
    vmax
        Maximum intensity shown in plot.
    ssps
        The grid object needed for plotting.
    flip
        If True, the plotted image is upside down. This is True by default, since it is more correct from a
        physical point of view.
    xlabel
        Determines whether x-axis is labeled or not.
    ylabel
        Determines whether y-axis is labeled or not.
    """
    # Plot distribution function.
    immap = plot_distribution_function(ax=ax, image=image, ssps=ssps, vmax=vmax, flip=flip, xlabel=xlabel,
                                       ylabel=ylabel)

    # Plot blobs
    insignificant_color = COLOR_OUTSIDE  # color for insignificant features
    feature_color = COLOR_INSIDE  # color for significant features -- inner part
    significant_color = COLOR_SIGNIFICANT  # color for significant features -- outer part.
    for blob in blobs:
        b = blob[0]
        c = blob[1]
        if c is None:
            y, x = b.position
            plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x, y), width=b.width, height=b.height,
                                color=insignificant_color, flip=flip)
        else:
            # feature is significant
            y1, x1 = b.position
            y2, x2 = c.position
            # If the width and height of the significant blob agree with map blob, we increase the former slightly for
            # better visualization.
            factor = 1.05
            plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x1, y1), width=b.width, height=b.height,
                                color=feature_color, flip=flip)
            plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x2, y2), width=factor * c.width,
                                height=factor * c.height, color=significant_color, linestyle="--", flip=flip)

    return immap

def plot_blobs(ax: Axes, image: np.ndarray, blobs: Sequence[GaussianBlob], vmax: float = None, ssps: MilesSSP = None,
               flip: bool = True, xlabel: bool = True, ylabel: bool = True):
    """
    Makes a blob-plot for given image and blobs.

    Parameters
    ----------
    ax
        The axis object where the plot should be created.
    image : shape (m, dim)
        The reference image.
    blobs
        The blobs.
    vmax
        Maximum intensity shown in plot.
    ssps
        The grid object needed for plotting.
    flip
        If True, the plotted image is upside down. This is True by default, since it is more correct from a
        physical point of view.
    xlabel
        Determines whether x-axis is labeled or not.
    ylabel
        Determines whether y-axis is labeled or not.
    """
    # First, plot distribution function.
    immap = plot_distribution_function(ax=ax, image=image, vmax=vmax, ssps=ssps, flip=flip, xlabel=xlabel,
                                       ylabel=ylabel)

    # Then, plot blobs.
    blob_color = COLOR_BLOB
    for blob in blobs:
        y, x = blob.position
        plot_scaled_ellipse(ax=ax, imshape=image.shape, center=(x, y), width=blob.width, height=blob.height,
                            color=blob_color, flip=flip)

    return immap

