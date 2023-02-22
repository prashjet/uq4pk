
from math import exp
import numpy as np
from scipy.special import iv


def bessel_kernel(t: float, n: int) -> np.ndarray:
    """
    Returns the one-dimensional normalized Bessel kernel for discrete analog of Gaussian:

    T(i, t) = exp(-t) * sum_{k=0}^infty 1 / (k!(m+k)!) * (t/2)^(2m + k).
    """
    assert t > 0.
    assert isinstance(n, (int, np.integer))
    assert n > 0

    # Make vector of indices.
    ind = np.arange(-n, n + 1)
    # Get values of Bessel function at those indices.
    i_t_vec = exp(-t) * iv(ind, t)

    assert i_t_vec.shape == (2 * n + 1, )
    return i_t_vec
