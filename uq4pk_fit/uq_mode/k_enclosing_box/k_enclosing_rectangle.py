
from copy import copy
import numpy as np
from typing import Tuple

from .s_structure import Structure1D
from .slab import Slab, divide_slab


EPS = 1e-15


class Rectangle:
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        """

        :param coords: Of shape (2, 2). First row is lower bound, second row is upper bound.
        """
        assert lb.shape == (2, )
        assert ub.shape == (2, )
        assert np.all(lb <= ub)
        self.ub = ub
        self.lb = lb
        self.area = (ub[0] - lb[0]) * (ub[1] - lb[1])


def k_enclosing_rectangle(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a set of n points in 2D, computes the smallest rectangle that contains k <= n points.
    This is an unoptimized implementation of the algorithm outlined in
        Chan and Har-Peled, "Smallest k-Enclosing Rectangle Revisited", 2019
        https://arxiv.org/pdf/1903.06785.pdf

    :param points: Of shape (n, 2).
    :return: lb, ub.
    """
    assert points.shape[1] == 2
    assert points.ndim == 2
    n = points.shape[0]
    assert k <= n
    # Initialize slabs tau and sigma such that they contain all points.
    y_max = points[:, 1].max()
    y_min = points[:, 1].min() - 1.     # To ensure that it also contains the lowest point.
    tau = Slab(y_min=y_min, y_max=y_max, points=points)
    sigma = Slab(y_min=y_min, y_max=y_max, points=points)
    # Start by computing any rectangle that contains at least k points.
    k_points = points[:k]
    lb = k_points.min(axis=0)
    ub = k_points.max(axis=0)
    rectangle = Rectangle(lb=lb, ub=ub)
    # Initialize data structure with all points marked.
    s = Structure1D(points=points, marked=np.full((n,), True), k=k)
    # Enter the recursion that computes the smallest rectangle.
    smallest_rectangle = divide_and_conquer(sigma=sigma, tau=tau, s=s, smallest_rectangle=rectangle)
    # Get lb, ub
    lb = smallest_rectangle.lb
    ub = smallest_rectangle.ub

    # Check that the result indeed contains k points.
    assert np.all(lb <= ub)
    k_in_rectangle = len([p for p in points if np.all(p >= lb) and np.all(p <= ub)])
    if k_in_rectangle < k:
        print(f"WARNING: Rectangle contains only {k_in_rectangle} < {k} points.")
    elif k_in_rectangle > k:
        # print(f"WARNING: Rectangle contains {k_in_rectangle} > {k} points.")
        pass

    return lb, ub


def divide_and_conquer(sigma: Slab, tau: Slab, s: Structure1D, smallest_rectangle: Rectangle) -> Rectangle:
    """
    Assume that sigma and tau each contain q points, that sigma is either completely above tau (in x2 coordinate)
    or tau = sigma.

    :param sigma: Of shape (r, 2).
    :param tau: Of shape (t, 2).
    :param s: Data structure.
    :return: the answer ???
    """
    if not s.enough_points:
        return smallest_rectangle
    elif sigma.num_in_slab == 1 and tau.num_in_slab == 1:
        # If there are less than k points inside, we cannot find a good rectangle.
        if not s.enough_inside:
            return smallest_rectangle
        l, u = s.shortest_interval()
        # Smallest rectangle.
        rectangle = Rectangle(lb=np.array([l, tau.y_max + EPS]), ub=np.array([u, sigma.y_min]))
        if rectangle.area < smallest_rectangle.area:
            smallest_rectangle = rectangle
    else:
        # Else, divide sigma into two horizontal subslabs, each containing q/2 points of P. Likewise divide tau.
        # then, recursively solve the problem for each sigma_i-tau_j pairing.
        tau_1, tau_2 = divide_slab(tau)
        sigma_1, sigma_2 = divide_slab(sigma)
        for sigma_i in [sigma_1, sigma_2]:
            for tau_j in [tau_1, tau_2]:
                assert sigma_i.num_in_slab >= 1
                assert tau_j.num_in_slab >= 1
                # Prepare s for recursion:
                # Create copy.
                s_new = copy(s)
                # Delete all points above sigma_i or below tau_j.
                s_new.delete(sigma=sigma_i, tau=tau_j)
                # Unmark all remaining points that are not in sigma_i or tau_j.
                s_new.unmark(sigma=sigma_i, tau=tau_j)
                smallest_rectangle = divide_and_conquer(sigma_i, tau_j, s_new, smallest_rectangle)

    return smallest_rectangle


