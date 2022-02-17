
from typing import List

from .filter_function import FilterFunction
from ..discretization import Partition


def direct_sum(ffunction_list: List[FilterFunction]):
    """
    Build the direct sum of a list of filter functions.

    :param ffunction_list:
        A list of LocalizationFunction objects.
    :return: The new localization function has dimension equal to the sum of the dimensions of the localization functions
        in lfunction_list, and size equal to the sum of the sizes. The underlying discretization is the join
         of all partitions.
    """
    # For every localization function, collect the dimension and the lists.
    dimension_sum = 0
    filter_list = []
    for ffunction in ffunction_list:
        ffunction_filters = ffunction.get_filter_list()
        # append the elements and filters to the merged element and filter list
        filter_list.extend(ffunction_filters)
        dimension_sum += ffunction.dim
    # Build a merged filter function from the merged discretization and the list of all filters
    merged_ffunction = FilterFunction(dim=dimension_sum, filter_list=filter_list)
    return merged_ffunction