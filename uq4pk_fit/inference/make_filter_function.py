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
    sigma1 = options.setdefault("sigma1", 1.)
    sigma2 = options.setdefault("sigma2", 1.)
    serious = options.setdefault("serious", False)
    boundary = options.setdefault("boundary", "zero")
    ffunction_f = uq_mode.GaussianFilterFunction2D(m=m_f, n=n_f, sigma1=sigma1, sigma2=sigma2, boundary=boundary)
    if dim_theta_v is None:
        # In the linear case, we are already done.
        ffunction_x = ffunction_f
        ffunction_vartheta = None
    else:
        # Create filter for theta_v
        ffunction_vartheta = uq_mode.IdentityFilterFunction(dim_theta_v)
        # combine the filter functions
        ffunction_x = uq_mode.direct_sum([ffunction_f, ffunction_vartheta])
    ffunction_theta = uq_mode.IdentityFilterFunction(dim=7)
    return ffunction_x, ffunction_f, ffunction_vartheta, ffunction_theta