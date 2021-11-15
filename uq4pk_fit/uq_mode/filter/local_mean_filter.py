"""
Contains class "LocalMeanFilter".
"""

import numpy as np

from ..filter import LinearFilter


class LocalMeanFilter(LinearFilter):
    """
    Defines the local mean filter that computes the local mean on a prescribed set of indices,
    with another set of indices as "localization window".
    """
    def __init__(self, indices: np.ndarray, window: np.ndarray):
        """
        :param indices: Defines the indices over which the local mean is computed.
        :param window: Defines the localization window. Note that it is required that "window" contains "indices".
        """
        # Assert that the entries of "indices" are a subset of the entries of "window".
        assert set(indices).issubset(set(window))
        # Find the relative coordinates of "indices" in "window".
        relative_indices = np.searchsorted(window, indices)
        # Make the weights: all indices in "indices" are weighted equally so that the weights sum to 1.
        # The other indices in "window" are not weighted at all.
        weights = np.zeros(window.size)
        weights[relative_indices] = 1 / relative_indices.size
        LinearFilter.__init__(self, indices=window, weights=weights)