
import numpy as np
from typing import List

from ..filter import LinearFilter
from .filter_function import FilterFunction
from ..partition.trivial_partition import TrivialPartition


class SimpleFilterFunction(FilterFunction):
    """
    Simple version of a filter function that associates to each coordinate a weighted sum of the complete parameter
    vector.
    """
    def __init__(self, dim: int, weights: List[np.ndarray]):
        """
        :param dim: The dimension of the underlying parameter space.
        :param weights: Each element in ``weights`` must be a numpy array of shape (``dim``,).
        """
        # make trivial partition
        partition = TrivialPartition(dim)
        # for each weight, make a corresponding filter
        filter_list = []
        for i in range(dim):
            filter_i = LinearFilter(indices=np.arange(dim), weights=weights[i])
            filter_list.append(filter_i)
        FilterFunction.__init__(self, partition, filter_list)
