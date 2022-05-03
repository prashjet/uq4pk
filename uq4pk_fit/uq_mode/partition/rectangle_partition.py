"""
Contains function "rectangle_partition"
"""

import numpy as np

from .image_partition import ImagePartition


def rectangle_partition(m, n, a, b) -> ImagePartition:
    """
    Makes a rectangle discretization.
    Returns indices for a discretization of an image of shape (n_x, n_y) into rectangles of shape (a_x, a_y).
    If the array is not evenly divisible, the rectangles may be larger.
    :param m: int
        Number of rows of the image.
    :param n: int
        Number of columns of the image.
    :param a: int
        Desired number of rows of each superpixel.
    :param b: int
        Desired number of columns for each superpixels.
    :return: list
        The i-th element of the list is a numpy array of ints, denoting the indices corresponding to the i-th superpixel.
    """
    # check input
    assert a < m, "a must be smaller than dim_y"
    assert b < n, "b must be smaller than dim"
    # create array of indices
    index_array = np.arange(m * n)
    index_array = np.reshape(index_array, (m, n))
    # compute x-split positions
    j = b
    x_split_positions = []
    while j + b <= n:
        x_split_positions.append(j)
        j += b
    i = a
    y_split_positions = []
    while i + a <= m:
        y_split_positions.append(i)
        i += a
    # initialize superpixel_list
    superpixel_list = []
    # now, split the array along y_axis
    rows = np.vsplit(index_array, y_split_positions)
    # then split along x_axis
    for row in rows:
        rectangles = np.hsplit(row, x_split_positions)
        # want flat index arrays in rectangles
        for arr in rectangles:
            arrflat = arr.flatten()
            superpixel_list.append(arrflat)
    partition = ImagePartition(m=m, n=n, superpixel_list=superpixel_list)
    return partition


