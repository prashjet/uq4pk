"""
Determines how sigma and scale are related. The relation is simply:

t_i = sigma_i^2.
"""

from math import sqrt
import numpy as np


def scale_to_sigma2d(t: float, r: float = 1.) -> np.ndarray:
    """
    Translates scale to sigma-parameter.

    :param t: Scale.
    :param r: Ratio of sigma[1] to sigma[0]
    :return: Sigma.
    """
    s = sqrt(t)
    sigma = np.array([r * s, s])
    return sigma


def sigma_to_scale2d(sigma: np.ndarray) -> float:
    """

    :param sigma:
    :return: Scale.
    """
    t = np.square(sigma[1])
    return t
