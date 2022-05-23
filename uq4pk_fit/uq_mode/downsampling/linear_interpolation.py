
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from typing import Tuple


def linear_interpolation(x: np.ndarray, shape: Tuple[int, int], indices: np.ndarray):
    """
    Given an image of shape (floor(m/a), floor(n/b)), creates an image of shape (m, n) by linear interpolation.

    :param x: The flattened vector of the downsampled image.
    :param rectangular_downsampling:
    :return: Of shape (m * n, ). The interpolated image (flat).
    """
    # Check that input is consistent.
    assert x.shape == (indices.size, )
    m, n = shape
    # Get grid of coordinates for im. A pixel (i,j) has coordinate (j + 0.5, m - x - 0.5).
    coords = indices_to_coords(indices, (m, n))
    # Apply linear interpolation. This yields a function.
    interpolation_function = LinearNDInterpolator(points=coords, values=x)
    # Get grid of coordinates for full image.
    all_coords = indices_to_coords(np.arange(m * n), shape=(m, n))
    # Note that x = x_1 and y = m - x[0]
    u = np.array([interpolation_function(x[0], x[1]) for x in all_coords]).flatten()
    # Perform sanity check.
    assert u.shape == (m * n, )
    # Return
    return u


def indices_to_coords(indices: np.ndarray, shape: Tuple[int, int]):
    """
    Given indices of pixels, returns coordinates.

    :param indices: Between 0 and m * n - 1.
    :param shape: (m, n).
    :return: Array of shape (indices.size, 2).
    """
    m, n = shape
    i_vals = indices // n
    j_vals = indices % n
    x_coords = j_vals + 0.5
    y_coords = m - i_vals - 0.5
    coords = np.column_stack([x_coords, y_coords])
    return coords