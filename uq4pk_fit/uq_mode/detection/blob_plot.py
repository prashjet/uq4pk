
from matplotlib import pyplot as plt
import numpy as np


def plot_blob(image, blobs):
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = ax.imshow(image, cmap="gnuplot", aspect="auto")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    scalecol = "r"
    rescol = "lime"
    for blob in blobs:
        if blob.size == 3:
            y, x, scale = blob
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * scale, color=scalecol, fill=False))
        else:
            y, x, scale, res = blob
            scalevar = scale - res + 0.1
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * scale, color=scalecol, fill=False))
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * scalevar, color=rescol, fill=False))