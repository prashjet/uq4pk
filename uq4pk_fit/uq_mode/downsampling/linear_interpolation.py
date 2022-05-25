
import numpy as np
from scipy.interpolate import interp2d
from typing import Tuple


def linear_interpolation(image: np.ndarray, shape: Tuple[int, int], a: int, b: int):
    """
    Given an image of shape (floor(m/a), floor(n/b)), creates an image of shape (m, n) by linear interpolation.

    :param image: The downsampled image.
    :param shape: The shape of the original image.
    :param a: Vertical downsampling.
    :param b: Horizontal downsampling.
    :return: Of shape (m, n). The interpolated image (flat).
    """
    # Check that input is consistent.
    m, n = shape
    # Get grid of sampled coordinates.
    i_indices = np.arange(0, m, a)
    j_indices = np.arange(0, n, b)
    x_sampled, y_sampled = indices_to_coords(i_indices, j_indices)
    # Apply linear interpolation. This yields a function.
    interpolation_function = interp2d(x=x_sampled, y=y_sampled, z=image, kind="linear")
    # Note that x = x_1 and y = m - x[0]
    all_i = np.arange(0, m)
    all_j = np.arange(0, n)
    all_x, all_y = indices_to_coords(all_i, all_j)
    u = interpolation_function(all_x, all_y)
    # Perform sanity check.
    assert u.shape == (m, n)
    # Return
    return u


def indices_to_coords(i: np.ndarray, j: np.ndarray):
    """
    Given indices of pixels, returns coordinates.

    :param indices: Between 0 and m * n - 1.
    :param shape: (m, n).
    :return: Array of shape (indices.size, 2).
    """
    x = j + 0.5
    y = i + 0.5
    return x, y