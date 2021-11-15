
import numpy as np
import numpy.typing as npt


class LinearFilter:
    """
    A LINEAR filter consists of a weight vector and an associated index set that defines the support
    of the filter functional (and its center).
    """
    def __init__(self, indices: np.ndarray, weights: np.ndarray):
        """
        :param indices: Array of ints. Defines the support of the filter, i.e. all indices that influence the filter
                value.
        :param weights: Defines the weights of each index in "indices". Therefore, the size must match that of
                "indices".
        """
        assert indices.size > 0
        assert weights.size == indices.size
        # sort the indices
        self.indices = indices
        self.weights = weights
        self._sort()
        self.size = self.indices.size   # the "size" of the localization functional is the length of its support

    def evaluate(self, v: np.ndarray) -> float:
        """
        Evaluates the filter. That is, it returns the value
            phi(v) = LinearFilter.weights @ v[LinearFilter.indices].
        :param v:
        :return: The value phi(v).
        """
        return self.weights @ v[self.indices]

    def extend(self, indices: npt.ArrayLike, weights: npt.ArrayLike):
        """
        Extends the filter by appending more indices with corresponding weights.
        :param indices: The indices that are appended to the filter.
        :param weights: The corresponding weights.
        :raises ValueError: If inputs are inconsistent.
        """
        # Check that indices are not contained in self.indices:
        if not set(indices).isdisjoint(self.indices):
            raise ValueError
        if indices.size != weights.size:
            raise ValueError
        self.indices = np.concatenate([self.indices, indices])
        self.weights = np.concatenate([self.weights, weights])
        # Sort and update size.
        self._sort()
        self.size += indices.size

    def shift(self, i: int):
        """
        Shifts all indices by a given amount.
        :param i:
        """
        self.indices += i

    def _sort(self):
        """
        Make sure that indices and weights are sorted.
        """
        sorting = np.argsort(self.indices)
        self.indices = self.indices[sorting]
        self.weights = self.weights[sorting]
