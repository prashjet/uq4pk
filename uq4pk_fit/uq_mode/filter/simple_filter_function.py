"""
Contains class "SimpleFilterFunction"
"""

import numpy as np

from ..filter import Filter
from .filter_function import FilterFunction
from ..partition.trivial_partition import TrivialPartition


class SimpleFilterFunction(FilterFunction):
    """
    Simple version of a filter function that associates to each coordinate a weighted sum of the complete parameter
    vector.
    """
    def __init__(self, dim, weights):
        """
        :param dim: int
        :param weights: list of length dim
            Each element in weights must be a (dim,) numpy array that determines the weights for each coordinate.
        """
        # make trivial partition
        partition = TrivialPartition(dim)
        # for each weight, make a corresponding filter
        filter_list = []
        for i in range(dim):
            filter_i = Filter(indices=np.arange(dim), weights=weights[i])
            filter_list.append(filter_i)
        FilterFunction.__init__(self, partition, filter_list)
