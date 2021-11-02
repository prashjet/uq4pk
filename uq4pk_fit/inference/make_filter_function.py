"""
Contains function "make_filter_function".
"""

import numpy as np

import uq4pk_fit.uq_mode as uq_mode


def make_filter_function(m_f, n_f, dim_theta_v=None, options: dict=None):
    """
    Defines the window function for the computation of LCIs in the case where theta_v is not fixed. In that case,
    each window also contains the parameter theta_v.
    """
    # Read/set options.
    kernel = options.setdefault("kernel", "laplace")
    a = options.setdefault("a", 1)
    b = options.setdefault("b", 1)
    c = options.setdefault("c", 6)
    d = options.setdefault("d", 6)
    h = options.setdefault("h", 3)
    serious = options.setdefault("serious", False)
    # Create filter function for f
    if kernel == "gauss":
        ffunction_f = uq_mode.SquaredExponentialFilterFunction(m=m_f, n=n_f, a=a, b=b, c=c, d=d, h=h)
    elif kernel == "laplace":
        ffunction_f = uq_mode.ExponentialFilterFunction(m=m_f, n=n_f, a=a, b=b, c=c, d=d, h=h)
    elif kernel == "mean":
        ffunction_f = uq_mode.ImageLocalMeans(m=m_f, n=n_f, a=a, b=b, c=c, d=d)
    elif kernel == "pixel":
        ffunction_f = uq_mode.PixelWithRectangle(m=m_f, n=n_f, a=a, b=b)
    elif kernel == "geometric":
        ffunction_f = uq_mode.GeometricFilterFunction(m=m_f, n=n_f, a=a, b=b, c=c, d=d)
    else:
        raise KeyError("Unknown kernel.")
    if dim_theta_v is None:
        # In the linear case, we are already done.
        ffunction_x = ffunction_f
        ffunction_vartheta = None
    else:
        # Create filter for theta_v
        ffunction_vartheta = uq_mode.IdentityFilterFunction(dim_theta_v)
        # combine the filter functions
        ffunction_x = uq_mode.direct_sum([ffunction_f, ffunction_vartheta])
        # Then, modify each f-filter such that it also includes all theta_v-indices, but not weighted.
        dim_f = m_f * n_f
        theta_v_indices = np.arange(dim_f, dim_f + dim_theta_v)
        theta_v_weights = np.zeros(dim_theta_v)
        for i in range(ffunction_f.size):
            ffunction_x.extend_filter(i, theta_v_indices, theta_v_weights)
        if serious:
            # And modify each theta-filter such that it includes all f-indices, but not weighted. (only in serious runs)
            f_indices = np.arange(dim_f)
            f_weights = np.zeros(dim_f)
            for i in range(ffunction_f.size, ffunction_x.size):
                ffunction_x.extend_filter(i, f_indices, f_weights)
    ffunction_theta = uq_mode.IdentityFilterFunction(dim=7)
    return ffunction_x, ffunction_f, ffunction_vartheta, ffunction_theta