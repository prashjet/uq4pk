"""
Contains function "samples_to_ci".
"""

import numpy as np

from ..filter import FilterFunction


def samples_to_ci(alpha, samples, ffunction: FilterFunction):
    """
    Given samples, computes the credible intervals according to a given filter function.
    :param samples: array_like, shape (M, N)
        The samples as an array. Each column corresponds to one sample.
    :param ffunction: FilterFunction
        The filter function that determines how the credible intervals are computed.
    :return: array_like, shape (M,2)
        The credible intervals. M is the dimension of the parameter space, and each row corresponds to the lower and
        upper bound for the credible interval for the i-th component.
    """
    # first, evaluate the filter function on each sample.
    n = samples.shape[1]
    phi_list = []
    for j in range(n):
        # Evaluate the filter function on the j-th sample and append to phi_list.
        phi_j = ffunction.evaluate(samples[:, j])
        # Enlarge
        phi_list.append(phi_j)
    # Turn phi_list into a numpy array.
    phi_arr = np.column_stack(phi_list)
    # sort each row
    phi_sorted = np.sort(phi_arr, axis=1)
    # cut off the alpha/2 smallest and alpha/2 largest values
    lower_cutoff = max(np.floor(0.5 * alpha * n - 1).astype(int), 0)
    upper_cutoff = min(np.ceil((1 - 0.5 * alpha) * n - 1).astype(int), n-1)
    lower_bounds = phi_sorted[:, lower_cutoff]
    upper_bounds = phi_sorted[:, upper_cutoff]
    credible_intervals = np.column_stack([lower_bounds, upper_bounds])
    return credible_intervals