
import numpy as np

from uq4pk_fit.filtered_credible_intervals.filter import FilterFunction
from ..k_enclosing_box import alpha_enclosing_box
from .fci_class import FCI


RTOL = 0.1  # Relative tolerance for credibility parameter.


def fci_sampling(alpha: float, samples: np.ndarray, ffunction: FilterFunction):
    """
    Computes a filtered credible interval from samples. That is, it computes vectors lb and ub such that 1-alpha of
    the samples satisfy lb <= ffunction(sample) <= ub. Note that this is different from requiring the samples to
    satify the same pointwise, i.e. lb_ij <= ffunction(sample)_ij <= ub_ij for 1-alpha of the samples, for all ij.

    Parameters
    ----------
    alpha
        The credibility parameter. Must satisfy 0 < alpha < 1.
        Example: alpha=0.05 corresponds to 95%-credibility.
    samples : shape (n, d)
        Here, n is the number of samples and d is the parameter space. This means that each row corresponds to a
        different sample.
    ffunction :
        The filter that is applied to the samples before computing the credible intervals.
    Returns
    -------
    fci : FCI
        Returns the computed filtered credible intervals as an `FCI` object.
    """
    # Check the input for consistency.
    assert 0 < alpha < 1
    assert samples.ndim == 2
    d = samples.shape[1]
    assert d == ffunction.dim

    # EVALUATE FILTER FUNCTION ON EACH SAMPLE
    n = samples.shape[0]
    phi_list = []
    for j in range(n):
        # Evaluate the filter function on the j-th sample and append to phi_list.
        phi_j = ffunction.evaluate(samples[j])
        phi_list.append(phi_j)
    filtered_samples = np.row_stack(phi_list)

    # Evaluate mean.
    filtered_mean = np.mean(filtered_samples, axis=0)

    # FIND SMALLEST BOX THAT CONTAINS (1 - alpha) OF SAMPLES.
    lb, ub = alpha_enclosing_box(alpha, points=filtered_samples)

    # Create FCI object.
    fci_obj = FCI(lower_stack=lb.reshape(1, -1), upper_stack=ub.reshape(1, -1), mean=filtered_mean)

    return fci_obj
