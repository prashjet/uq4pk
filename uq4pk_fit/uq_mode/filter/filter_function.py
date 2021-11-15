
import copy
import numpy as np
import typing

from ..partition import Partition
from .linearfilter import LinearFilter


class FilterFunction:
    """
    A filter function maps elements of a given partition to individual LinearFilter-objects, and thus determines how the
    parameter space is filtered in the computation of the filtered credible intervals.
    """
    def __init__(self, partition: Partition, filter_list: typing.List[LinearFilter]):
        """
        :param partition: The partition from which the filter function maps.
        :param filter_list: A list of the same size as "partition".
            The i-th element of "filter_list" will be mapped to the i-th partition element.
        """
        self._partition = copy.deepcopy(partition)
        self.dim = partition.dim
        # check that for each partition element there is a localization functional
        assert partition.size == len(filter_list)
        self.size = partition.size
        self._filter_list = filter_list.copy()

    def change_filter(self, i: int, filter: LinearFilter):
        """
        Changes the filter to which the i-th partition element is associated.
        :param i: Index of the partition element to be remapped.
        :param filter: The new LinearFilter object to which the i-th partition element is mapped.
        """
        self._filter_list[i] = filter

    def extend_filter(self, i: int, indices: np.ndarray, weights: np.ndarray):
        """
        Extends the i-th filter.
        :param i: Index of the filter to be extended.
        :param indices: The additional indices, as integer array.
        :param weights: The associated weights. The length of the ``weights`` parameter must equal the length of
        :raises: ValueError
        """
        self._filter_list[i].extend(indices, weights)

    def enlarge(self, v: np.ndarray) -> np.ndarray:
        """
        Given a vector of size FilterFunction.size, returns a vector of size self.dim
        by setting the i-th partition element equal to v[i].
        :param v: Must satisfy k == :py:attr:`size`
        :return: The enlarged vector.
        """
        v_enlarged = np.zeros(self.dim)
        for i in range(self.size):
            v_enlarged[self._partition.element(i)] = v[i]
        return v_enlarged

    def evaluate(self, v: np.ndarray) -> np.ndarray:
        """
        Evaluates the filter function at v. That is, it returns a new vector w, where
            w[i] = FilterFunction.filter(i).evaluate(v).
        NOTE: This function could probably be optimized, as it uses an inefficient for-loop. However, it is not
        called often, and hence it does really matter.
        :param v: Must satisfy n == :py:attr:`dim`
        :return: Of shape (:py:attr:`size`,).
        """
        w = np.zeros(self.size)
        for i in range(self.size):
            filter_i = self._filter_list[i]
            w[i] = filter_i.evaluate(v)
        return w

    def get_element_list(self) -> typing.List[np.ndarray]:
        """
        Returns all the elements of the underlying partition as list.
        :return: list
        """
        return copy.deepcopy(self._partition.get_element_list())

    def get_filter_list(self) -> typing.List[LinearFilter]:
        """
        Returns list of all filters.
        :return:
        """
        return copy.deepcopy(self._filter_list)

    def filter(self, i: int) -> LinearFilter:
        """
        Returns the filter associated to the i-th partition element.
        :param i: index
        :return:
        """
        return self._filter_list[i]
