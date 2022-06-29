
import numpy as np
from typing import Tuple

from .k_enclosing_rectangle import k_enclosing_rectangle


class Box:

    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        assert np.all(lb <= ub)
        self.dim = lb.size
        assert lb.shape == (self.dim, ) == ub.shape
        self.lb = lb
        self.ub = ub

    @property
    def volume(self) -> float:
        diff = self.ub - self.lb
        volume = np.prod(diff)
        return volume


def smallest_interval_1d(points: np.ndarray, k: int) -> Tuple[float, float]:
    """
    Given points in 1 d, computes the smallest interval that contains k points.

    :param points:
    :param k:
    :return: lb, ub
    """
    m = points.size
    assert points.shape == (m, )
    assert m >= k
    # Sort points.
    points_sorted = np.sort(points)
    # If there are exactly k points, simply return minimum and maximum.
    if m == k:
        lb = points_sorted[0]
        ub = points_sorted[-1]
    else:
        # Then, compute the smallest interval that contains k of the i coordinates.
        add = points_sorted[k - 1:]
        sub = points_sorted[:-k + 1]
        distances = add - sub
        i_min = np.argmin(distances)
        lb = points_sorted[i_min]
        ub = points_sorted[i_min + k - 1]

    assert lb <= ub
    return lb, ub


def shrink_on_dimension(points: np.ndarray, box: Box, i: int, k: int):
    """
    Given a d-1 dimensional box that contains >= k projections of the points, create a smallest box
    that contains k points by multiplying it with the smallest interval.
    :param points: Of shape (n, d).
    :param box: Box of dimension d-1.
    :param i: Index of remaining dimension.
    :return: box of dimension d, containing k points.
    """
    # Get points in box.
    s_i = np.delete(points, (i), axis=1)
    mask_in_box = np.all((s_i[:] >= box.lb) & (s_i[:] <= box.ub), axis=1)
    points_in_box = points[mask_in_box]
    # Get i coordinates of these points.
    i_coords = points_in_box[:, i]
    # Then, compute the smallest interval that contains k of the i coordinates.
    lb_i, ub_i = smallest_interval_1d(points=i_coords, k=k)
    # Finally, the new box is the Cartesian product of this interval with the given (d-1)-dimensional box.
    lb_d = np.insert(box.lb, i, lb_i)
    ub_d = np.insert(box.ub, i, ub_i)
    box_d = Box(lb_d, ub_d)

    return box_d


def k_enclosing_box(points: np.ndarray, k: int) -> Box:
    """
    Recursively computes a d-dimensional k-enclosing box, using the recursion described in
        Segal and Kedem, "Enclosing k points in the smallest axis parallel rectangle", 1998

    :param points: Of shape (n, d), where n is the number of points.
    :param k: Must satisfy k <= n.

    :return: lb, ub
    """
    assert points.ndim == 2
    n, d = points.shape
    assert k <= n

    # If d > 2, we reduce to couple of (d-1)-dimensional problems.
    if d > 2:
        # Create candidate for smallest box.
        lb = points.min(axis=0)
        ub = points.max(axis=0)
        smallest_box = Box(lb=lb, ub=ub)
        # Project points set on hyperplanes.
        for i in range(d):
            s_i = np.delete(points, (i), axis=1)
            # For this point set, find all (d-1)-dimensional boxes that contain k to n points.
            for j in range(k, n):
                box_j = k_enclosing_box(s_i, k=j)
                # Then, use the i-th axis to bound exactly k points of S.
                completed_box_j = shrink_on_dimension(points, box_j, i, k)
                # If the resulting box is smaller than the currently smallest box, replace.
                if completed_box_j.volume < smallest_box.volume:
                    smallest_box = completed_box_j
    else:
        # For d = 2, we can compute using the k-enclosing rectangle method.
        lb, ub = k_enclosing_rectangle(points, k)
        smallest_box = Box(lb, ub)

    return smallest_box





