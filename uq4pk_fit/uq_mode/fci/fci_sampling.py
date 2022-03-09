
import numpy as np

from uq4pk_fit.uq_mode.filter import FilterFunction
from .fci import FCI
from .k_enclosing_box import alpha_enclosing_box


RTOL = 0.1  # Relative tolerance for credibility parameter.


def fci_sampling(alpha: float, samples, ffunction: FilterFunction) -> FCI:
    """
    Computes a filtered credible interval from samples. That is, it computes vectors lb and ub such that 1-alpha of
    the samples satisfy lb <= ffunction(sample) <= ub. Note that this is different from requiring the samples to
    satify the same pointwise, i.e. lb_ij <= ffunction(sample)_ij <= ub_ij for 1-alpha of the samples, for all ij.

    :param alpha: The credibility parameter. Must satisfy 0 < alpha < 1.
        Example: alpha=0.05 corresponds to 95%-credibility.
    :param samples: Of shape (n, d), where n is the number of samples and d is the parameter space. This means that
        each row corresponds to a different sample.
    :param ffunction: The filter function that determines how the credible intervals are computed. An object of type
        uq4pk_fit.uq_mode.filter.FilterFunction.
    :return: An FCI object.
    """
    # Check the input for consistency.
    assert 0 < alpha < 1
    assert samples.ndim == 2
    assert samples.shape[1] == ffunction.dim

    # EVALUATE FILTER FUNCTION ON EACH SAMPLE
    n = samples.shape[0]
    phi_list = []
    for j in range(n):
        # Evaluate the filter function on the j-th sample and append to phi_list.
        phi_j = ffunction.evaluate(samples[j])
        phi_list.append(phi_j)
    filtered_samples = np.row_stack(phi_list)

    # FIND SMALLEST BOX THAT CONTAINS (1 - alpha) OF SAMPLES.
    credible_box = alpha_enclosing_box(alpha=alpha, points=filtered_samples)

    # Create FCI object.
    fci_obj = FCI(phi_lower_enlarged=credible_box[0], phi_upper_enlarged=credible_box[1])
    return fci_obj