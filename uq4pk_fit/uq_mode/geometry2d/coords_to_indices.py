"""
Contains function "coords_to_indices".
"""

import numpy as np


def coords_to_indices(m, n, coords):
    """
    Given the (x,y)-coordinates of a pixel in the image, returns the number of that pixel.
    :param coords: (2, j) array of ints
        Rows correspond to coordinates, columns correspond to pixels.
    :return: (j,) array of ints
    """
    assert coords.shape[0] == 2
    if coords.ndim == 1:
        # if j=0, reshape to (2,1)-array
        j = 1
        coords = np.reshape(coords, (2, 1))
    else:
        j = coords.shape[1]
    index = n * coords[0, :] + coords[1, :]
    # assert that none of the indices is outside the array bounds
    if not (np.all(index >= 0) and np.all(index <= m * n - 1)):
        print("BAD ")
    assert index.size == j
    return index