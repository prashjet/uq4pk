"""
Contains class "SquaredExponentialFilterFunction".
"""

from math import exp
from .distance_filter_function import DistanceFilterFunction


class SquaredExponentialFilterFunction(DistanceFilterFunction):
    """
    Special case of ImageFilterFunction that associates to each pixel a correspondingly positioned exponential filter.
    See also uq_mode.fci.ExponentialFilter.
    """
    def __init__(self, m: int, n: int, a: int, b: int, c: int, d: int, h: float = 1.):
        self._h = h
        DistanceFilterFunction.__init__(self, m, n, a, b, c, d, h)

    def _weighting(self, d: float) -> float:
        """
        Exponential weighting function:
        w(d) = 2^(- h * d ** 2).
        :param d: The distance of a given pixel to the center.
        :return: The weight w(d).
        """
        w = exp(- d ** 2)
        return w
