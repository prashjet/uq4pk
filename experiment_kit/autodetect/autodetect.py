
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import laplace
from skimage.feature import peak_local_max

from uq4pk_fit.special_operators import DiscreteLaplacian


def autodetect(image: np.ndarray, scale: float, savename: str):
    """
    Automatically detects features at different scales.

    :param image: The image.
    :param scale: The scale at which the image was filtered.
    :param savename: The name of the file where the image should be saved.
    """
    # Detect local maxima
    coordinates = peak_local_max(image, min_distance=1, threshold_rel=0.01)
    # Plot the image, with circles at the local maxima
    ax = plt.axes()
    plt.imshow(image)
    for i in range(coordinates.shape[0]):
        x_coord = coordinates[i, 1]
        y_coord = coordinates[i, 0]
        ax.add_patch(plt.Circle((x_coord, y_coord), scale, color='r',
                                   fill=False))
    # Save the image in pyplot
    plt.savefig(savename, bbox_inches="tight")



