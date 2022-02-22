import numpy as np

from .linear_filter import LinearFilter
from .filter_function import FilterFunction


class SimpleFilter(LinearFilter):
    """
    Simple filter defined by weight vector.
    """
    def __init__(self, weights: np.ndarray):
        assert weights.ndim == 1
        self.dim = weights.size
        self.weights = weights.copy()


class SimpleFilterFunction(FilterFunction):
    """
    Simple filter function consisting of simple filters.
    """
    def __init__(self, weights: np.ndarray):
        """
        :param weights: (n, n)-array, where the i-th row corresponds to the weight vector for the i-th filter.
        """
        assert weights.ndim == 2
        assert weights.shape[0] == weights.shape[1]
        filter_list = []
        for w in weights:
            filter_w = SimpleFilter(weights=w)
            filter_list.append(filter_w)
        FilterFunction.__init__(self, dim=weights.shape[0], filter_list=filter_list)

