
from math import sqrt, log
import numpy as np
from scipy.linalg import eigh


def is_credible(x, alpha, xmap, costfun):
    """
    Checks whether a given vector lies in the (1-alpha) posterior credible region.
    :param x: ndarray
        The vector for which we want to know whether it lies in the Pereyra credible region.
    :param alpha: float > 0
        The credibility parameter.
    :param xmap: ndarray
        The maximum-a-posteriori estimate. x and xmap must habe same size.
    :param cost: function
        The maximum-a-posteriori cost function.
    :return: bool
        True, if x lies in the posterior credible region. Otherwise False.
    """
    assert x.size == xmap.size
    map_cost = costfun(xmap)
    n = xmap.size
    tau = sqrt(16 * log(3/alpha) / n)
    if costfun(x) <= map_cost + n * (tau + 1):
        return True
    else:
        return False

def credible_region(alpha, H, y, delta, xbar, xmap):
    """
    Computes the level that determines the (1-alpha) Pereyra credible region
    ||A(x-h)||_2^2 <= lvl for the linear Gaussian model
    Y = H @ X + V,
    V ~ normal(0, delta^2*Identity),
    X ~ normal(xbar, Identity).
    returns A, z, lvl
    """
    n = xmap.size
    tau = sqrt(16 * log(3/alpha) / n)
    map_cost = 0.5 * ((np.linalg.norm(H @ xmap - y)/delta)**2 + np.linalg.norm(xmap - xbar)**2)
    lvl = map_cost + n * (tau + 1)
    # completing the squares
    H_delta = H / delta
    y_delta = y / delta
    A = 0.5 * (H_delta.T @ H_delta + np.identity(n))
    c = 0.5 * (y_delta @ y_delta + xbar @ xbar)
    b = -H_delta.T @ y_delta - xbar
    s, U = eigh(A)
    A_sqrt_inv = U * np.divide(1, np.sqrt(s))
    A_inv = A_sqrt_inv @ A_sqrt_inv.T
    h = - 0.5 * A_inv @ b
    k = c - 0.25 * b.T @ A_inv @ b
    lvl -= k
    return A_sqrt_inv, h, lvl

