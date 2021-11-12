"""
Contains class "PixelWithRectangle"
"""


import numpy as np

from .image_filter_function import ImageFilterFunction
from ..filter import LinearFilter
from .geometry2d import rectangle_indices, indices_to_coords
from ..partition.trivial_image_partition import TrivialImagePartition


class PixelWithRectangle(ImageFilterFunction):
    """
    For a rectangular image, this filter functions simply associates to each pixel its own value, but uses
    a rectangular localization window.
    """
    def __init__(self, m, n, a, b):
        """
        :param m: int
            Number of rows of the image.
        :param n: int
            Number of columns of the image.
        :param a: int
            Vertical width of the localization window.
        :param b: int
            Horizontal width of the localization window.  For example, if a=1 and b=2, then each window will be a
            3x5 rectangle centered at the active pixel.
        """
        self._a = a
        self._b = b
        self.m = m
        self.n = n
        self.dim = m * n
        # setup trivial image partition
        partition = TrivialImagePartition(m=m, n=n)
        # now, setup rectangular windows around each pixel
        window_list = self._setup_windows()
        # from each window, make the corresponding localization functional by weighting only the active pixel:
        filter_list = []
        for i in range(self.m * self.n):
            window_i = window_list[i]
            # find the index i in window_i:
            j_i = np.where(window_i == i)[0][0]
            # make the weight vector where only the index j_i is weighted
            weights_i = np.zeros(window_i.size)
            weights_i[j_i] = 1.
            lfunctional_i = LinearFilter(indices=window_i, weights=weights_i)
            filter_list.append(lfunctional_i)
        ImageFilterFunction.__init__(self, image_partition=partition, filter_list=filter_list)

    def _setup_windows(self):
        """
        Creates a list of windows around each pixel.
        :return: list
            A list of numpy arrays of the same length as 'self.windows'.
        """
        windows = []
        for i in range(self.dim):
            # get x-y coordinates associated to the active index
            index_coords = indices_to_coords(m=self.m, n=self.n, indices=i)
            # determine the upper-left and lower-right corner of the window
            upper_left = index_coords[:, 0] - np.array([self._b, self._a])
            lower_right = index_coords[:, 0] + np.array([self._b, self._a])
            # get indices of the window using the coordinates for its upper left and lower right corner
            window_i = rectangle_indices(self.m, self.n, upper_left, lower_right)
            windows.append(window_i)
        return windows





