"""
Contains class "GeometricFilterFunction"
"""


from .distance_filter_function import DistanceFilterFunction


class GeometricFilterFunction(DistanceFilterFunction):
    """
    Special case of ImageFilterFunction that associates to each pixel a correspondingly positioned exponential filter.
    See also uq_mode.fci.ExponentialFilter.
    """
    def __init__(self, m, n, a, b, c, d, p=1, k=1.):
        self._p = p
        self._k = k
        DistanceFilterFunction.__init__(self, m, n, a, b, c, d)

    def _weighting(self, d):
        """
        Exponential weighting function:
        w(d) = k / (1+d)^p
        :param d: float
            The distance of a given pixel to the center.
        :return: float
            The weight w(d).
        """
        w = self._k / ((1 + d) ** self._p)
        return w