"""
Contains class "FilterFunction".
"""

import copy
import numpy as np
import typing

from ..partition import Partition
from .filter import Filter


class FilterFunction:
    """
    A filter function maps elements of a given partition to individual Filter-objects, and thus determines how the
    parameter space is filtered in the computation of the filtered credible intervals.
    """
    def __init__(self, partition: Partition, filter_list: typing.List[Filter]):
        """
        :param partition: uq_mode.partition.Partition
            The partition from which the filter function maps.
        :param filter_list: list
            A list of the same size as "partition".
            The i-th element of "filter_list" will be mapped to the i-th partition element.
        """
        self._partition = copy.deepcopy(partition)
        self.dim = partition.dim
        # check that for each partition element there is a localization functional
        assert partition.size == len(filter_list)
        self.size = partition.size
        self._filter_list = filter_list.copy()

    def change_filter(self, i, filter: Filter):
        """
        Changes the filter to which the i-th partition element is associated.
        :param i: int
            Number of the partition element to be remapped.
        :param filter: uq_mode.fci.Filter
            The new Filter object to which the i-th partition element is mapped.
        """
        self._filter_list[i] = filter

    def extend_filter(self, i, indices, weights):
        """
        Extends the i-th filter.
        :param i: int
        :param indices: array_like, int, shape (K,)
        :param weights: array_like, int, shape (K,)
        :raises: ValueError
        """
        self._filter_list[i].extend(indices, weights)

    def enlarge(self, v):
        """
        Given a vector of size FilterFunction.size, returns a vector of size self.dim
        by setting the i-th partition element equal to v[i].
        :param v: array_like, shape (K,)
            Must satisfy k == LocalizationFunction.size
        :return: array_like, shape (N,)
            The enlarged vector.
        """
        v_enlarged = np.zeros(self.dim)
        for i in range(self.size):
            v_enlarged[self._partition.element(i)] = v[i]
        return v_enlarged

    def evaluate(self, v):
        """
        Evaluates the filter function at v. That is, it returns a new vector w, where
            w[i] = FilterFunction.filter(i).evaluate(v).
        NOTE: This function could probably be optimized, as it uses an inefficient for-loop. However, it is not
        called often, and hence it does really matter.
        :param v: (n,) vector
            Must satisfy n == FilterFunction.dim.
        :return: (FilterFunction.size,) numpy vector
        """
        w = np.zeros(self.size)
        for i in range(self.size):
            filter_i = self._filter_list[i]
            w[i] = filter_i.evaluate(v)
        return w

    def get_element_list(self):
        """
        Returns all the elements of the underlying partition as list.
        :return: list
        """
        return copy.deepcopy(self._partition.get_element_list())

    def get_filter_list(self):
        """
        Returns list of all filters.
        :return: list
        """
        return copy.deepcopy(self._filter_list)

    def filter(self, i):
        """
        Returns the filter associated to the i-th partition element.
        :param i: int
        :return: Filter
        """
        return self._filter_list[i]
