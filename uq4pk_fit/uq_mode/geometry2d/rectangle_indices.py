"""
Contains auxiliary functions for geometry:
    - "rectangle_indices"
    - "coords_to_indices"
    - "indices_to_coords"
"""

import numpy as np

from .coords_to_indices import coords_to_indices
from .indices_to_coords import indices_to_coords


def rectangle_indices(m, n, upper_left, lower_right, return_relative=False):
    """
    Given the coordinates of the upper-left and lower-right corner of a rectangle inside an (m,dim)-image, returns
    the coordinates of all points that are inside that rectangle.
    :param m: int
        Number of rows of the image.
    :param n: int
        Number of columns of the image.
    :param upper_left: (2,) numpy array of ints
    :param lower_right: (2,) numpy array of ints
    :param return_relative: bool
        If True, the function also returns the relative indices of the cut-lci_vs_fci rectangle within the original one.
        For example, if upper_left = [-1, -1] and lower_right = [3,3], then the function will also return the indices
        of the 4x4 square that are inside the lower right 3x3 square that is inside the image.
    :return: list
        Returns a list of (2,)-numpy arrays corresponding to all coordinates of the rectangle inside the image.
    """
    indices_inside, upper_left_inside, lower_right_inside = _compute_indices_inside(m, n, upper_left, lower_right)
    if return_relative:
        # get size of the rectangle
        m_rect, n_rect = lower_right - upper_left + 1
        # get relative position of upper_left_inside and lower_right_inside
        upper_left_relative = upper_left_inside - upper_left
        lower_right_relative = lower_right_inside - upper_left
        # get indices using the _get_rectangle_indices function
        indices_relative, _, _ = _compute_indices_inside(m=m_rect, n=n_rect, upper_left=upper_left_relative,
                                                   lower_right=lower_right_relative)
        assert indices_relative.size == indices_inside.size     # sanity check
        return indices_inside, indices_relative
    else:
        return indices_inside


# PROTECTED

def _compute_indices_inside(m, n, upper_left, lower_right):
    # Remove the parts of the rectangle that lie outside the image.
    upper_left_inside = np.array([max(upper_left[0], 0), max(upper_left[1], 0)])
    lower_right_inside = np.array([min(lower_right[0], m-1), min(lower_right[1], n-1)])
    # Get a list of all coordinates inside the image.
    all_coords = indices_to_coords(m, n, np.arange(m * n))
    # Remove all coordinates outside rectangle
    in_rectangle = np.all(
        (upper_left_inside[:, np.newaxis] <= all_coords) & (lower_right_inside[:, np.newaxis] >= all_coords), axis=0)
    rectangle_coords = all_coords[:, in_rectangle]
    # Translate the coordinates to indices
    rectangle_indices = coords_to_indices(m=m, n=n, coords=rectangle_coords)
    # Return the indices
    return rectangle_indices, upper_left_inside, lower_right_inside