
import numpy as np
from .filter_function import FilterFunction, LinearFilter
from ..partition import *


class IdentityFilterFunction(FilterFunction):
    def __init__(self, dim):
        partition = TrivialPartition(dim=dim)
        filter_list = []
        for i in range(dim):
            weights_i = np.zeros(dim)
            weights_i[i] = 1.
            filter_i = LinearFilter(indices=np.arange(dim), weights=weights_i)
            filter_list.append(filter_i)
        FilterFunction.__init__(self, partition=partition, filter_list=filter_list)
