
import numpy as np

from .fci import fci_sampling
from .filter import MarginalizingFilterFunction


def marginal_ci_from_samples(alpha: float, samples: np.ndarray, axis: int):
    """
    Computes simultaneous credible intervals for 2-dimensional samples, marginalized to a given axis.
    For example, if we are given (m, n)-dimensional samples and we want the first axis, then this returns a credible
    interval in R^m.

    Parameters
    ----------
    alpha : between 0 and 1
        The credibility parameter. For example, `alpha=0.05` corresponds to 95%-credibility.
    samples : shape (k, m, n)
        Samples from which the marginal credible intervals should be computed.
    axis :
        The axis on which the samples should be projected. Must be 0 or 1.

    Returns
    -------
    lb : (d, )
        The lower bound of the credible intervals. Its dimension `d` is equal to `samples.shape[axis + 1]`.
    ub : (d, )
        The corresponding upper bound.
    """
    assert axis in [0, 1]
    assert samples.ndim == 3
    k, m, n = samples.shape
    # Get flattened samples (of shape (k, m * n)).
    flattened_samples = samples.reshape(k, m * n)
    # Create Marginalizing filter for given axis
    filter_map = MarginalizingFilterFunction(shape=(m, n), axis=axis)
    # Compute corresponding FCI and append to list.
    fci_obj = fci_sampling(alpha=alpha, samples=flattened_samples, ffunction=filter_map)
    lb = fci_obj.lower
    mean = fci_obj.mean
    ub = fci_obj.upper

    return lb, mean, ub
