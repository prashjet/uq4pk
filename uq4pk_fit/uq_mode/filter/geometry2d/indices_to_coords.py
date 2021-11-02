"""
Contains function "indices_to_coords".
"""

import numpy as np


def indices_to_coords(m, n, indices):
    """
    Returns the x- and y-coordinates of the i-th pixel in a (m,n)-image
    :param m: int
        Number of rows of the image.
    :param n: int
        Number of columns of the image.
    :param indices: (j,) array of ints
        List of indices.
    :return: (2, j) array
        The i-th column corresponds to the x- and y- coordinate of the pixel with index 'indices[j]'.
    """
    x_coord = indices // n
    y_coord = indices % n
    coords = np.row_stack((x_coord, y_coord))
    return coords