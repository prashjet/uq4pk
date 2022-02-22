
import copy
import numpy as np
from typing import List

from .linear_filter import LinearFilter


class FilterFunction:
    """
    A filter function maps each coordinate to a LinearFilter object.
    """
    dim: int    # The dimension of the underlying space.

    def __init__(self, dim: int, filter_list: List[LinearFilter]):
        # Check that each filter has the right dimension.
        for filter in filter_list:
            assert filter.dim == dim
        self.dim = dim
        self._filters = filter_list

    def filter(self, i: int) -> LinearFilter:
        """
        Returns the filter for the i-th coordinate.
        """
        return self._filters[i]

    def change_filter(self, i: int, filter: LinearFilter):
        """
        Changes the filter to which the i-th discretization element is associated.
        :param i: Index of the discretization element to be remapped.
        :param filter: The new LinearFilter object to which the i-th discretization element is mapped.
        """
        self._filters[i] = filter

    def evaluate(self, v: np.ndarray) -> np.ndarray:
        """
        Evaluates the filter function at v. That is, it returns a new vector w, where
            w[i] = FilterFunction.filter(i).evaluate(v).
        NOTE: This function could probably be optimized, as it uses an inefficient for-loop. However, it is not
        called often, and hence it does really matter.
        :param v: Must satisfy n == :py:attr:`dim`
        :return: Of shape (:py:attr:`size`,).
        """
        w = np.zeros(self.dim)
        for i in range(self.dim):
            filter_i = self._filters[i]
            w[i] = filter_i.evaluate(v)
        return w

    def get_filter_list(self) -> List[LinearFilter]:
        """
        Returns list of all filters.
        :return:
        """
        return copy.deepcopy(self._filters)
