
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import numpy as np

from uq4pk_fit.special_operators import DiscreteLaplacian
from skimage.feature import blob_log, peak_local_max


def autodetect(image: np.ndarray, scale: float, savename: str):
    """
    Automatically detects features at different scales. To do this, applies a Laplacian filter

    :param image: The image.
    :param scale: The scale at which the image was filtered.
    :param savename: The name of the file where the image should be saved.
    """
    # Create the matching Laplacian operator
    m, n = image.shape
    laplacian = DiscreteLaplacian(m=m, n=n)
    # Apply the Laplacian to the image.
    filtered_image = laplacian
    # Detect local maxima
    coordinates = peak_local_max(image, min_distance=1, threshold_rel=0.01)
    # Plot the image, with circles at the local maxima
    plt.imshow(image)
    for i in range(coordinates.shape[0]):
        x_coord = coordinates[i, 1]
        y_coord = coordinates[i, 0]
        plt.axis.add_patch(plt.Circle((x_coord, y_coord), scale, color='r',
                                   fill=False))
    # Save the image in pyplot
    plt.savefig(savename, bbox_inches="tight")



