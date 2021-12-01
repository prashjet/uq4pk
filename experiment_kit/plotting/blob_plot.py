
from matplotlib import pyplot as plt
import numpy as np


def blob_plot(image: np.ndarray, blobs, savename: str, resolution: float = None,
              vmin: float = None, vmax: float = None):
    """
    Makes a blob-plot for given image an blobs.

    :param image: The image as (m, n)-array.
    :param blobs: The blobs. Must be a 2-dimensional array of shape (k, 3), where each row (s, i, j) corresponds to
        a blob of scale s at position (i, j).
    :param savename: The name under which the plot should be saved (including file-type).
    :param vmin:
    :param vmax:
    """
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = ax.imshow(image, cmap="gnuplot", aspect="auto", vmin=vmin, vmax=vmax)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    if resolution is None:
        scalecol = "lime"
        rescol = "y"
    else:
        scalecol = "r"
        rescol = "lime"
    for blob in blobs:
        if blob.size == 3:
            y, x, scale = blob
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * scale, color=scalecol,
                                    fill=False))
        else:
            y, x, scale, resolution = blob
            min_area = scale - resolution + 0.1
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * min_area, color=rescol,
                                fill=False))
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * scale, color=scalecol,
                                    fill=False))
    plt.savefig(savename, bbox_inches="tight")