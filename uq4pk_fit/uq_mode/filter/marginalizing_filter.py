
import numpy as np
from typing import Tuple

from .image_filter_function import ImageFilterFunction
from .linear_filter import LinearFilter


class MarginalizingFilter(LinearFilter):
    """
    Defines marginalizing filter for images, given by the functional
    phi_i(x) =
    """
    def __init__(self, shape: Tuple[int, int], axis: int, i: int):
        """

        :param shape: The shape of the image.
        :param axis: The axis over which we want to sum.
        :param i: The parameter at which we sum.
        """
        assert axis in [0, 1]
        m, n = shape
        self.dim = m * n
        if axis == 0:
            assert 0 <= i < n
        else:
            assert 0 <= i < m
        # We want the weight vector to have 1's at the positions that are multiplied with the i-th column/row of the
        # image. We can obtain such a vector by creating a zero image of the correct shape, filling the i-th column/row
        # with 1's, and then flatten it.
        weight_image = np.zeros(shape)
        if axis == 0:
            weight_image[:, i] = 1
        else:
            weight_image[i, 1] = 1
        LinearFilter.__init__(self, weights=weight_image.flatten())


class MarginalizingFilterFunction(ImageFilterFunction):
    """
    Defines a marginalizing filter. Given an image of shape (m, n), the user can specify an axis.
    The resulting filter function is then the result of marginalizing that axis.
    As an example, given axis=0, the resulting filter function is then of size n, where each filter is given as
    phi_j(x) = sum_i x[i][j], j in [n].
    """
    def __init__(self, shape: Tuple[int, int], axis: int):
        assert len(shape) == 2
        m, n = shape
        assert m >= 1 and n >= 1
        assert axis in [0, 1]
        other_axis = int(axis == 0)
        # Create list of filters.
        filter_list = [MarginalizingFilter(shape, axis, i) for i in range(shape[other_axis])]
        # Call constructor of ImageFilterFunction.
        ImageFilterFunction.__init__(self, m=m, n=n, filter_list=filter_list)

