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
        # Then, modify each f-filter such that it also includes all theta_v-indices, but not weighted.
        dim_f = m_f * n_f
        theta_v_indices = np.arange(dim_f, dim_f + dim_theta_v)
        theta_v_weights = np.zeros(dim_theta_v)
        for i in range(ffunction_f.dim):
            ffunction_x.extend_filter(i, theta_v_indices, theta_v_weights)
        if serious:
            # And modify each theta-filter such that it includes all f-indices, but not weighted. (only in serious runs)
            f_indices = np.arange(dim_f)
            f_weights = np.zeros(dim_f)
            for i in range(ffunction_f.dim, ffunction_x.size):
                ffunction_x.extend_filter(i, f_indices, f_weights)
    ffunction_theta = uq_mode.IdentityFilterFunction(dim=7)
    return ffunction_x, ffunction_f, ffunction_vartheta, ffunction_theta