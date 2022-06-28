"""
Visualizes the ideas behind window-localization and two-level localization.
"""


from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import numpy as np

from uq4pk_fit.uq_mode.geometry2d import rectangle_indices, indices_to_coords


m = 24
n = 24

center = np.array([9, 9])

# Reference image is simply the zero image.
im_ref = np.zeros((m, n))


def create_window_grid(upper_left: np.ndarray, lower_right: np.ndarray):
    """
    Returns the full grid inside the given window.
    :param upper_left:
    :param lower_right:
    :return:
    """
    window_indices = rectangle_indices(m=m, n=n, upper_left=upper_left, lower_right=lower_right)
    # Translate them to coordinates
    window_coords = indices_to_coords(m=m, n=n, indices=window_indices)
    # Translate the coords to a meshgrid.
    # Compute vertical size of window.
    s1 = lower_right[0] - upper_left[0]
    s2 = lower_right[1] - upper_left[1]
    x = window_coords[0]
    x = x.reshape((s1+1, s2+1))
    y = window_coords[1]
    y = y.reshape((s1+1, s2+1))
    return x, y


# Visualize localization window.
def plot_localization_window(ax: Axes, w1: int, w2: int):

    # Use the information in `window` to build a meshgrid.
    # Get the indices of the rectangle
    diagonal = np.array([w1, w2])
    upper_left = center - diagonal
    lower_right = center + diagonal + 1   # +1, so that we also get the right and lower boundaries of the window.
    x, y = create_window_grid(upper_left, lower_right)

    # mark the active pixel
    image = im_ref
    image[center[0], center[1]] = 1.

    # plot the image
    ax.imshow(im_ref)
    #plot the grid
    segs1 = np.stack((x-0.5, y-0.5), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1))
    ax.add_collection(LineCollection(segs2))
    ax.set_xticks([])
    ax.set_yticks([])


# Visualize two-level discretization.
def plot_twolevel_discretization(ax: Axes, w1: int, w2: int, d1: int, d2: int):
    # Create the basic grid.
    x, y = np.meshgrid(np.arange(0, n+d2, d2), np.arange(0, m+d1, d1))

    # Create the window grid.
    diagonal = np.array([w1 * d1, w2 * d2])
    upper_left = center - diagonal
    lower_right = center + diagonal + np.array([d1, d2])
    x_window, y_window = create_window_grid(upper_left, lower_right)

    # mark the active pixel
    image = im_ref
    image[center[0], center[1]] = 1.

    # plot the image
    ax.imshow(im_ref)
    # plot the base grid
    segs1 = np.stack((x - 0.5, y - 0.5), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1))
    ax.add_collection(LineCollection(segs2))
    # plot the window grid
    segs1 = np.stack((x_window - 0.5, y_window - 0.5), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1))
    ax.add_collection(LineCollection(segs2))
    ax.set_xticks([])
    ax.set_yticks([])

