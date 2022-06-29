import numpy as np

from .slab import Slab


class Structure1D:
    """
    Implements the 1d-data structure necessary for the k-enclosing rectangle.
    """
    def __init__(self, points: np.ndarray, marked: np.ndarray, k: int):
        """

        :param points: Of shape (n, 2). Points in R.
        :param marked: Boolean mask of shape (n, 1). Determines which points count as marked and which don't.
        :param k:
        """
        # Sort points from left to right.
        n = points.shape[0]
        assert n >= k
        assert points.shape == (n, 2)
        assert marked.shape == (n,)
        self._k = k
        # Points are stored according to x-coordinates.
        ascending_x_coord = np.argsort(points[:, 0])
        self._points = points[ascending_x_coord]
        self._marked = marked.copy()

    @property
    def enough_inside(self):
        enough = (self._num_inside >= self._k)
        return enough

    @property
    def enough_points(self) -> bool:
        """
        Returns true if the number of points is at least k.
        """
        num_stored = self._points.shape[0]
        enough = (num_stored >= self._k)
        return enough

    def shortest_interval(self):
        """
        Returns the shortest interval that contains k points.

        :return: lb, ub
            lb: Lower bound of interval.
            ub: Upper bound of interval.
        """
        # If there are exactly k points, then simply return the complete interval.
        if self._num_inside == self._k:
            lb = self._points[0, 0]
            ub = self._points[-1, 0]
        else:
            # Compute distance vector.
            distance_vector = self._compute_distance_vector()
            # Return minimum of k-th diagonal.
            #  Get k-th diagonal.
            i = np.argmin(distance_vector)
            x_points = self._points[~self._marked, 0]
            # Minimizing interval is p[i+k-1] - p[i].
            lb = x_points[i]
            ub = x_points[i + self._k - 1]
        assert lb <= ub
        return lb, ub

    def delete(self, sigma: Slab, tau: Slab):
        """
        Delete all marked points above sigma or below tau.
        """
        # Create mask that is True for all points to be deleted.
        above_sigma = sigma.points_above(self._points)
        below_tau = tau.points_below(self._points)
        to_be_deleted = (above_sigma | below_tau) & self._marked
        self._points = self._points[~to_be_deleted]
        # Deleted points are also unmarked.
        self._marked = self._marked[~to_be_deleted]

    def unmark(self, sigma: Slab, tau: Slab):
        """
        Unmarks all points that are not inside sigma or tau.

        :param below:
        :param above:
        """
        inside_sigma = sigma.points_inside(self._points)
        inside_tau = tau.points_inside(self._points)
        inside_sigma_or_tau = inside_sigma | inside_tau
        self._marked = inside_sigma_or_tau

    def _compute_distance_vector(self):
        """
        Computes vector with entries p_{i + k - 1} - p_i, for i = 0,..., n-k + 1.
        :param points: Of shape (n - k, s).
        """
        unmarked_points = self._points[~self._marked]
        unmarked_x_values = unmarked_points[:, 0]
        add = unmarked_x_values[self._k - 1:]
        sub = unmarked_x_values[:-self._k + 1]
        distance_vector = add - sub
        return distance_vector

    @property
    def _num_inside(self):
        points_inside = self._points[~self._marked]
        num = points_inside.shape[0]
        return num





