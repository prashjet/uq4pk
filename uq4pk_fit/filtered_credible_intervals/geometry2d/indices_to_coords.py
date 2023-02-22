"""
Contains function "indices_to_coords".
"""

import numpy as np


def indices_to_coords(m: int, n: int, indices: np.ndarray):
    """
    Returns the x- and y-coordinates of the i-th pixel in a (m,dim)-image
    """
    x_coord = indices // n
    y_coord = indices % n
    coords = np.row_stack((x_coord, y_coord))
    # If j = 1, reshape to vector
    if coords.shape[1] == 1:
        coords = coords.reshape((2, ))
    return coords
