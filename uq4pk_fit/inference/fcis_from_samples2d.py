

import numpy as np
from typing import Sequence

from uq4pk_fit.uq_mode import fci_sampling, GaussianFilterFunction2D


def fcis_from_samples2d(alpha: float, samples: np.ndarray, sigmas: Sequence[np.ndarray]):
    """
    Computes stack of FCIs for different scales.

    :param alpha: The credibility parameter.
    :param samples: Of shape (k, m, n), where n is the number of samples and (m, n) their shape as image
    :param sigmas: List of sigmas for Gaussian filter.
    :return: lower_stack, upper_stack
        - lower_stack: Of shape (k, m, n).
        - upper_stack: Of shape (k, m, n).
    """
    assert samples.ndim == 3
    k, m, n = samples.shape
    # Get flattened samples (of shape (k, m * n)).
    flattened_samples = samples.reshape(k, m * n)
    # Iteratively apply fci_sampling for all scale-values.
    lower_list, upper_list = [], []
    for sigma in sigmas:
        # Create GaussianFilter for given value of sigma.
        filter = GaussianFilterFunction2D(m=m, n=n, sigma=sigma, boundary="reflect")
        # Compute corresponding FCI and append to list.
        fci_obj = fci_sampling(alpha=alpha, samples=flattened_samples, ffunction=filter)
        lower_list.append(fci_obj.lower.reshape(m, n))
        upper_list.append(fci_obj.upper.reshape(m, n))
    # Make stacks from lists.
    lower_stack = np.array(lower_list)
    upper_stack = np.array(upper_list)

    return lower_stack, upper_stack
