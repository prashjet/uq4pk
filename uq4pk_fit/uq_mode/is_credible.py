
from math import sqrt, log

def is_credible(x, alpha, x_map, costfun):
    """
    Checks whether a given vector lies in the approximate (1-alpha)-credible region obtained from the
    Pereyra approximation.
    :param x: ndarray
        The vector for which we want to know whether it lies in the Pereyra credible region.
    :param alpha: float > 0
        The credibility parameter.
    :param x_map: ndarray
        The maximum-a-posteriori estimate. x and xmap must habe same size.
    :param cost: function
        The maximum-a-posteriori cost function.
    :return: bool
        True, if x lies in the posterior credible region. Otherwise False.
    """
    assert x.size == x_map.size
    map_cost = costfun(x_map)
    n = x_map.size
    tau = sqrt(16 * log(3/alpha) / n)
    if costfun(x) <= map_cost + n * (tau + 1):
        return True
    else:
        return False

