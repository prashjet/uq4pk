

import numpy as np
from typing import Sequence

from .fci import fci_sampling
from .filter import BesselFilterFunction2D, IdentityFilterFunction


def fcis_from_samples2d(alpha: float, samples: np.ndarray, sigmas: Sequence[np.ndarray]):
    """
    Computes stack of FCIs for different scales.

    Parameters
    ---
    alpha
        The credibility parameter.
    samples : shape (k, m, n)
        n is the number of samples and (m, n) their shape as image
    sigmas :
        List of standard deviations for Gaussian filter.

    Returns
    ---
    lower_stack : shape (k, m, n)
    upper_stack : shape (k, m, n)
    """
    assert samples.ndim == 3
    k, m, n = samples.shape
    # Get flattened samples (of shape (k, m * n)).
    flattened_samples = samples.reshape(k, m * n)
    # Iteratively apply fci_sampling for all scale-values.
    lower_list, upper_list = [], []
    for sigma in sigmas:
        # Create GaussianFilter for given value of sigma.
        filter_map = BesselFilterFunction2D(m=m, n=n, sigma=sigma)
        # Compute corresponding FCI and append to list.
        fci_obj = fci_sampling(alpha=alpha, samples=flattened_samples, ffunction=filter_map)
        # Check whether filtered mean lies inside interval.
        lower_list.append(fci_obj.lower.reshape(m, n))
        upper_list.append(fci_obj.upper.reshape(m, n))
    # Make stacks from lists.
    lower_stack = np.array(lower_list)
    upper_stack = np.array(upper_list)

    return lower_stack, upper_stack


def pcis_from_samples2d(alpha: float, samples: np.ndarray):
    """
    Computes pixelwise credible intervals.

    Parameters
    ---
    alpha
        The credibility parameter.
    samples : shape (k, m, n),
        n is the number of samples and (m, n) their shape as image.

    Returns
    ---
    lower : shape (m, n)
    upper : shape (m, n)
    """
    assert samples.ndim == 3
    k, m, n = samples.shape
    # Get flattened samples (of shape (k, m * n)).
    flattened_samples = samples.reshape(k, m * n)
    filter_map = IdentityFilterFunction(dim=m * n)
    fci_obj = fci_sampling(alpha=alpha, samples=flattened_samples, ffunction=filter_map)

    return fci_obj.lower, fci_obj.upper
