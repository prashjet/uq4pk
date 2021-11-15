
from typing import List

from .filter_function import FilterFunction
from ..partition.partition import Partition


def direct_sum(ffunction_list: List[FilterFunction]):
    """
    Build the direct sum of a list of filter functions.

    :param ffunction_list:
        A list of LocalizationFunction objects.
    :return: The new localization function has dimension equal to the sum of the dimensions of the localization functions
        in lfunction_list, and size equal to the sum of the sizes. The underlying partition is the join
         of all partitions.
    """
    # For every localization function, collect the dimension and the lists.
    dimension_sum = 0
    element_list = []
    filter_list = []
    for ffunction in ffunction_list:
        # The partition elements and the indices of the filters have to be shifted by dimension_sum
        ffunction_elements = ffunction.get_element_list()
        ffunction_filters = ffunction.get_filter_list()
        for element, filter in zip(ffunction_elements, ffunction_filters):
            element += dimension_sum
            filter.shift(dimension_sum)
        # append the elements and filters to the merged element and filter list
        element_list.extend(ffunction_elements)
        filter_list.extend(ffunction_filters)
        dimension_sum += ffunction.dim
    # Build a merged partition
    merged_partition = Partition(dim=dimension_sum, elements=element_list)
    # Build a merged filter function from the merged partition and the list of all filters
    merged_ffunction = FilterFunction(partition=merged_partition, filter_list=filter_list)
    return merged_ffunction