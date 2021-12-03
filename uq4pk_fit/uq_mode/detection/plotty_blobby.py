
from matplotlib import pyplot as plt
import numpy as np
from typing import List


def plotty_blobby(image: np.ndarray, blobs: List[np.ndarray], scales: List[float]):
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    im = ax.imshow(image, cmap="gnuplot", aspect="auto")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    scalecol = "lime"
    rescol = "r"
    for blob in blobs:
        if blob.shape == (3, ):
            iscale, y, x = blob
            sigma = np.sqrt(2 * scales[iscale])
            ax.add_patch(plt.Circle((x, y), np.sqrt(2) * sigma, color=scalecol, fill=False))
        else:
            y1, x1, s1 = blob[0]
            ax.add_patch(plt.Circle((x1, y1), np.sqrt(2) * s1, color=scalecol, fill=False))
            y2, x2, s2 = blob[1]
            ax.add_patch(plt.Circle((x2, y2), np.sqrt(2) * s2, color=rescol, fill=False))