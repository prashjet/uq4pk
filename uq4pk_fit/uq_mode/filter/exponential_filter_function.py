"""
Contains class "ExponentialFilterFunction".
"""

from math import exp
from .distance_filter_function import DistanceFilterFunction


class ExponentialFilterFunction(DistanceFilterFunction):
    """
    Special case of ImageFilterFunction that associates to each pixel a correspondingly positioned exponential filter.
    See also uq_mode.fci.ExponentialFilter.
    """
    def __init__(self, m, n, a, b, c, d, h=1.):
        self._h = h
        DistanceFilterFunction.__init__(self, m, n, a, b, c, d)


    def _weighting(self, d):
        """
        Exponential weighting function:
        w(d) = 2^(-d).
        :param d: float
            The distance of a given pixel to the center.
        :return: float
            The weight w(d).
        """
        w = exp(- d / self._h)
        return w

