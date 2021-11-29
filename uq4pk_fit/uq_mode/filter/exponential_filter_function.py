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
    def __init__(self, m: int, n: int, a: int, b: int, c: int, d: int, h: float = 1.):
        """

        :param m: Number of rows of the image.
        :param n: Number of columns of the image.
        :param a: Number of rows of each superpixel.
        :param b: Number of columns of each superpixel.
        :param c: Vertical width of the localization frame around each superpixel.
        :param d: Horizontal width of the localization frame around each superpixel.
        :param h: Scaling parameter.
        """
        self._h = h
        DistanceFilterFunction.__init__(self, m, n, a, b, c, d, 2 * h)


    def _weighting(self, d: float) -> float:
        """
        Exponential weighting function:
        w(d) = 2^(-d / h).
        :param d: The distance of a given pixel to the center.
        :returns: The value w(d).
        """
        w = exp(- d)
        return w

