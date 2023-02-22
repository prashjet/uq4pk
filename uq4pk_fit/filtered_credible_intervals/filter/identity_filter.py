
import numpy as np

from .linear_filter import LinearFilter
from .filter_function import FilterFunction


class EvaluationFilter(LinearFilter):
    """
    The simple filter x -> x[i].
    """

    def __init__(self, dim: int, i: int):
        weights = np.zeros(dim)
        weights[i] = 1.
        LinearFilter.__init__(self, weights)


class IdentityFilterFunction(FilterFunction):
    def __init__(self, dim: int):
        filter_list = []
        for i in range(dim):
            filter_i = EvaluationFilter(dim=dim, i=i)
            filter_list.append(filter_i)
        FilterFunction.__init__(self, dim=dim, filter_list=filter_list)

