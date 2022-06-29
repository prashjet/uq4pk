
from copy import copy
import numpy as np
from typing import Tuple


class Slab:
    """
    A slab is a horizontal stripe defined through the condition (y_min < y <= y_max).
    """
    def __init__(self, y_min: float, y_max: float, points: np.ndarray):
        self.y_min = y_min
        self.y_max = y_max
        self.points = points

    @property
    def num_in_slab(self) -> int:
        """
        Returns the number of points in the slab.
        """
        y_points = self._y_points_in_slab()
        return y_points.size

    @property
    def y_half(self) -> float:
        """
        Returns the y-value that would halve the slab by splitting <= y_half, > y_half
        """
        y_vals = np.sort(self._y_points_in_slab())
        n_in_slab = y_vals.size
        n_half = np.ceil(n_in_slab / 2).astype(int) - 1
        y_half = y_vals[n_half]
        return y_half

    def points_inside(self, points: np.ndarray):
        """
        Takes a set of points and returns a mask that is True if a points is inside the slab.
        :param points: Of shape (m, 2).
        :return: Of shape (m, 2). Boolean mask.
        """
        mask_inside = (points[:, 1] > self.y_min) & (points[:, 1] <= self.y_max)
        return mask_inside

    def points_above(self, points: np.ndarray):
        """
        Takes a set of points and returns a mask that is True if a points is above (>) the slab.
        :param points: Of shape (m, 2).
        :return: Of shape (m, 2). Boolean mask.
        """
        mask_above = (points[:, 1] > self.y_max)
        return mask_above

    def points_below(self, points: np.ndarray):
        """
        Takes a set of points and returns all points that are below (<=) the slab.
        :param points: Of shape (m, 2).
        :return: Of shape (m, 2). Boolean mask.
        """
        mask_below = (points[:, 1] <= self.y_min)
        return mask_below

    def _y_points_in_slab(self):
        y_points = self.points[:, 1]
        points_in_slab = y_points[(y_points > self.y_min) & (y_points <= self.y_max)]
        return points_in_slab



def divide_slab(slab: Slab) -> Tuple[Slab, Slab]:
    """
    Horizontally divides a slab into two slabs containing the same number of points (+-1).
    If the slab contains only 1 point, two copies are returned.

    :param slab: Of shape (r, 2).
    :return: slab1, slab2
    """
    assert isinstance(slab, Slab)
    r = slab.num_in_slab
    if r == 1:
        # If the slab contains only one point, we return two copies of it.
        slab1 = copy(slab)
        slab2 = copy(slab)
    else:
        # Find the y-value that halves the slab.
        y_half = slab.y_half
        slab1 = Slab(y_min=y_half, y_max=slab.y_max, points=slab.points)
        slab2 = Slab(y_min=slab.y_min, y_max=y_half, points=slab.points)
    return slab1, slab2