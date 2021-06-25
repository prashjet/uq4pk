
import numpy as np


def partition(n_x, n_y, a_x, a_y):
    """
    Returns indices for a partition of an image of shape (n_x, n_y) into rectangles of shape (a_x, a_y).
    If the array is not evenly divisible, the rectangles may be larger.
    :param n_x: int
    :param n_y: int
    :param a_x: int
    :param a_y: int
    :return:
    """
    # check input
    assert a_x < n_x, "a_x must be smaller than n_x"
    assert a_y < n_y, "a_y must be smaller than n_y"
    # create array of indices
    index_array = np.arange(n_x * n_y)
    index_array = np.reshape(index_array, (n_x, n_y))
    # compute x-split positions
    i = a_x
    x_split_positions = []
    while(i+a_x <= n_x):
        x_split_positions.append(i)
        i += a_x
    j = a_y
    y_split_positions = []
    while(j+a_y <= n_y):
        y_split_positions.append(j)
        j += a_y
    # initialize partition as list
    partition = []
    # now, split the array along x_axis
    rows = np.vsplit(index_array, x_split_positions)
    # then split along y_axis
    for row in rows:
        rectangles = np.hsplit(row, y_split_positions)
        # want flat index arrays in rectangles
        for arr in rectangles:
            arrflat = arr.flatten()
            partition.append(arrflat)
    return partition


