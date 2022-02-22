"""
Contains class "Partition".
"""

import numpy as np


class Partition:
    """
    A discretization is a list of index sets that discretization the given space.
    """
    def __init__(self, dim, elements):
        """
        :param dim: int
            The dimension of the partitioned space.
        :param elements: list
            A list of numpy arrays. Each array must only contains integers between 0 and dim-1. The elements must be
            disjoint.
        """
        self.dim = dim
        self._check_elements(elements)
        self._element_list = elements
        # Create the index, i.e. the list that maps each coordinate to the index of the corresponding partition element.
        self._index = -1 * np.ones(dim, dtype=int)
        for i in range(len(elements)):
            element_i = elements[i]
            self._index[element_i] = i
        self.size = len(elements)

    def element(self, i):
        """
        Returns the i-th element.
        :param i: int
        :return: numpy array of ints
        """
        return self._element_list[i]

    def in_which_element(self, i) -> int:
        """
        Returns the index of the element that contains the i-th coordinate.
        """
        return self._index[i]

    def get_element_list(self):
        """
        Returns the list of all elements
        :return: list
        """
        return self._element_list.copy()

    # PROTECTED

    def _check_elements(self, elements):
        # check that each element only contains integers between 0 and dim-1.
        for element in elements:
            assert np.all(0 <= element) and np.all(element < self.dim)
        # Then check that the elements are disjoint by merging them and checking for repetition.
        all_indices = np.concatenate(elements)
        no_duplicates = (len(np.unique(all_indices)) == len(all_indices))
        assert no_duplicates, "Partition must consist of disjoint elements!"

