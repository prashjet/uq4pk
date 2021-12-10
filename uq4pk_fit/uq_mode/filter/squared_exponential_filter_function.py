"""
Contains class "SquaredExponentialFilterFunction".
"""

from math import exp
import numpy as np
from typing import Literal, Union

from .distance_filter_function import DistanceFilterFunction


class SquaredExponentialFilterFunction(DistanceFilterFunction):
    """
    Special case of ImageFilterFunction that associates to each pixel a correspondingly positioned exponential filter.
    See also uq_mode.fci.ExponentialFilter.
    """
    def __init__(self, m: int, n: int, a: int, b: int, c: int, d: int, h: Union[float, np.ndarray] = 1.,
                 boundary: Literal["reflect", "zero"] = "reflect"):
        self._h = h
        DistanceFilterFunction.__init__(self, m, n, a, b, c, d, h=np.sqrt(4 * h), boundary=boundary)

    def _weighting(self, d: float) -> float:
        """
        Exponential weighting function:
        w(d) = 2^(- d ** 2).
        :param d: The distance of a given pixel to the center.
        :return: The weight w(d).
        """
        w = exp(- d ** 2)
        return w
