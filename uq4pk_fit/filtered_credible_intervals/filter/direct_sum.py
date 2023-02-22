import numpy as np

from .filter_function import FilterFunction
from .simple_filter_function import SimpleFilterFunction


def direct_sum(ffunction1: FilterFunction, ffunction2: FilterFunction):
    """
    Build the direct sum of a list of filter functions.
    """
    dim1 = ffunction1.dim
    dim2 = ffunction2.dim
    # Initialize weights list.
    weights_list = []
    # Get the filters for the first ffunction
    filters1 = ffunction1.get_filter_list()
    # Expand each filter with zero weights.
    zero_weights = np.zeros(dim2)
    for filter in filters1:
        filter.extend(zero_weights)
        weights_list.append(filter.weights)
    # Getthe filters of the second ffunction
    filters2 = ffunction2.get_filter_list()
    # Expand each filter with zero weights (before).
    zero_weights = np.zeros(dim1)
    for filter in filters2:
        filter.extend(zero_weights, before=True)
        weights_list.append(filter.weights)
    weights = np.array(weights_list)
    # Build a merged filter function from the merged discretization and the list of all filters
    merged_ffunction = SimpleFilterFunction(weights=weights)
    return merged_ffunction