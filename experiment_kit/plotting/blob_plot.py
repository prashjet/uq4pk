
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from typing import List, Tuple


def blob_plot(image: np.ndarray, blobs: List[Tuple], savename: str, vmin: float = None, vmax: float = None,
              scaling: np.ndarray = None):
    """
    Makes a blob-plot for given image an blobs.

    :param image: The image as (m, n)-array.
    :param blobs: The blobs. Each blob is given as a tuple (b, c), where b is the MAP-blob and c is either None
        or the significant blob. (A blob is an array of format (s, i, j), where s is the scale and (i, j) the center
        index for the blob.
    :param savename: The name under which the plot should be saved (including file-type).
    :param vmin:
    :param vmax:
    """
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = ax.imshow(image, cmap="gnuplot", aspect="auto", vmin=vmin, vmax=vmax)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    insignificant_color = "red"     # color for insignificant features
    feature_color = "yellow"    # color for significant features -- inner part
    significant_color = "lime"  # color for significant features -- outer part.
    for blob in blobs:
        b = blob[0]
        c = blob[1]
        if c is None:
            # feature is not significant
            s_x, s_y, y, x = b
            w = 2 * sqrt(2) * s_x
            h = 2 * sqrt(2) * s_y
            ax.add_patch(patches.Ellipse((x, y), width=w, height=h, color=insignificant_color,
                            fill=False))
        else:
            # feature is significant
            s_x1, s_y1, y1, x1 = b
            s_x2, s_y2, y2, x2 = c
            w1 = 2 * sqrt(2) * s_x1
            h1 = 2 * sqrt(2) * s_y1
            w2 = 2 * sqrt(2) * s_x2
            h2 = 2 * sqrt(2) * s_y2
            ax.add_patch(patches.Ellipse((x1, y1), width=w1, height=h1, color=feature_color,
                                         fill=False))
            ax.add_patch(patches.Ellipse((x2, y2), width=w2, height=h2, color=significant_color,
                                         fill=False))
    plt.savefig(savename, bbox_inches="tight")