
import numpy as np

from uq4pk_fit.uq_mode.filter import FilterFunction
from .fci import FCI


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

    # first, evaluate the filter function on each sample.
    n = samples.shape[0]
    phi_list = []
    for j in range(n):
        # Evaluate the filter function on the j-th sample and append to phi_list.
        phi_j = ffunction.evaluate(samples[j])
        phi_list.append(phi_j)
    phi_arr = np.row_stack(phi_list)

    # sort each row
    phi_sorted = np.sort(phi_arr, axis=0)

    # Find multidimensional FCI via bisection.
    gamma_high = alpha
    gamma_low = 0.
    gamma = gamma_high
    while gamma_high - gamma_low > RTOL * alpha:
        # cut off the alpha/2 smallest and alpha/2 largest values
        lower_cutoff = max(np.floor(0.5 * gamma * n - 1).astype(int), 0)
        upper_cutoff = min(np.ceil((1 - 0.5 * gamma) * n - 1).astype(int), n-1)
        lower_bounds = phi_sorted[lower_cutoff]
        upper_bounds = phi_sorted[upper_cutoff]
        credible_intervals = np.column_stack([lower_bounds, upper_bounds])

        # Check that FCI property is satisfied.
        mask = np.all(phi_arr >= lower_bounds, axis=1) & np.all(phi_arr <= upper_bounds, axis=1)
        masked_array = phi_arr[mask, :]
        n_samples_in_fci = masked_array.shape[0]
        ratio = n_samples_in_fci / n
        if ratio >= 1 - alpha:
            gamma_low = gamma
            gamma = 0.5 * (gamma_high + gamma_low)
        else:
            gamma_high = gamma
            gamma = 0.5 * (gamma_high + gamma_low)

    # Compute with respect to gamma_low to ensure that ratio >= 1 - alpha
    # cut off the alpha/2 smallest and alpha/2 largest values
    lower_cutoff = max(np.floor(0.5 * gamma_low * n - 1).astype(int), 0)
    upper_cutoff = min(np.ceil((1 - 0.5 * gamma_low) * n - 1).astype(int), n - 1)
    lower_bounds = phi_sorted[lower_cutoff]
    upper_bounds = phi_sorted[upper_cutoff]

    # Create FCI object.
    fci_obj = FCI(phi_lower_enlarged=lower_bounds, phi_upper_enlarged=upper_bounds)
    return fci_obj