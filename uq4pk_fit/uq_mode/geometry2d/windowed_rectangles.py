"""
Contains function "windowed_rectangles".
"""

import numpy as np

from ..discretization import rectangle_partition
from .indices_to_coords import indices_to_coords
from .rectangle_indices import rectangle_indices


def windowed_rectangles(m, n, a, b, c, d):
    """
    Creates a list of framed rectangles.
    :param m: int
    :param n: int
    :param a: int
    :param b: int
    :param c: int
    :param d: int
    :return: ImagePartition, list
        Returns the discretization of rectangles together with the list of windows.
    """
    rectangles = rectangle_partition(m, n, a, b)
    # for each rectangle, obtain the corresponding framing window.
    window_list = []
    for rectangle in rectangles.get_element_list():
        # get x-y coordinates of all indices in 'window'
        rectangle_coords = indices_to_coords(m, n, rectangle)
        # determine the upper-left and lower-right corner of the frame
        upper_left = rectangle_coords[:, 0] - np.array([c, d])
        lower_right = rectangle_coords[:, -1] + np.array([c, d])
        # get indices of the window using the coordinates for its upper left and lower right corner
        window = rectangle_indices(m=m, n=n, upper_left=upper_left, lower_right=lower_right)
        # append to "window_list"
        window_list.append(window)
    return rectangles, window_list