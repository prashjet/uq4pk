
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max


def autodetect(image: np.ndarray, scale: float, vmax: float, savename: str, ticks):
    """
    Automatically detects features at different scales.

    :param image: The image.
    :param scale: The scale at which the image was filtered.
    :param vmax: The maximum.
    :param savename: The name of the file where the image should be saved.
    """
    # Add zero boundary to image
    m, n = image.shape
    int_scale = np.ceil(scale).astype(int)
    patched_image = np.zeros((m + 2 * int_scale, n + 2 * int_scale))
    patched_image[int_scale:-int_scale, int_scale:-int_scale] = image
    # Detect local maxima
    coordinates = peak_local_max(patched_image, min_distance=int_scale, threshold_abs=0.01 * vmax)
    # Translate the coordinates to coordinates of the original image
    coordinates = coordinates - int_scale
    # Plot the image, with circles at the local maxima
    cmap = plt.get_cmap("gnuplot")
    fig = plt.figure(figsize=(6, 2.5))
    ax = plt.axes()
    for i in range(coordinates.shape[0]):
        x_coord = coordinates[i, 1]
        y_coord = coordinates[i, 0]
        ax.add_patch(plt.Circle((x_coord, y_coord), scale, color='r',
                                   fill=False))
    im = plt.imshow(image, vmax=vmax, vmin=0., cmap=cmap, aspect="auto")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("density")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Metallicity [Z/H]")
    # Manage ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Ticks don't work with circles.
    #t_ticks, z_ticks, img_t_ticks, img_z_ticks = ticks
    #ax.set_xticks(img_t_ticks)
    #ax.set_xticklabels(t_ticks)
    #ax.set_yticks(img_z_ticks)
    #ax.set_yticklabels(z_ticks)
    # Save the image in pyplot
    plt.savefig(savename, bbox_inches="tight")



