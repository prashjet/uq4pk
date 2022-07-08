
import numpy as np

from uq4pk_fit.uq_mode import fci_sampling, MarginalizingFilterFunction


def marginal_ci_from_samples(alpha: float, samples: np.ndarray, axis: int):
    """
    Computes marginalized credible intervals for given samples.

    :param alpha: The credibility parameter.
    :param samples: Of shape (k, m, n), where n is the number of samples and (m, n) their shape as image
    :return: lb, ub
        - lb: Of shape (k, ).
        - mean: Of shape (k, ).
        - ub: Of shape (k, ).
    """
    assert axis in [0, 1]
    assert samples.ndim == 3
    k, m, n = samples.shape
    # Get flattened samples (of shape (k, m * n)).
    flattened_samples = samples.reshape(k, m * n)
    # Create Marginalizing filter for given axis
    filter = MarginalizingFilterFunction(shape=(m, n), axis=axis)
    # Compute corresponding FCI and append to list.
    fci_obj = fci_sampling(alpha=alpha, samples=flattened_samples, ffunction=filter)
    lb = fci_obj.lower
    mean = fci_obj.mean
    ub = fci_obj.upper

    return lb, mean, ub