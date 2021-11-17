
import numpy as np
from typing import List

from .image_filter_function import ImageFilterFunction
from ..filter import LinearFilter
from .geometry2d import find_indices, rectangle_indices, indices_to_coords
from ..partition.trivial_image_partition import TrivialImagePartition


class ImageLocalMeans(ImageFilterFunction):
    """
    A filter function that associates to each superpixel the corresponding local means with a localization frame of
    prescribed width.
    """

    def __init__(self, m: int, n: int, a:int, b: int, c:int, d: int):
        """
        :param m: Number of rows of the image.
        :param n: Number of columns of the image.
        :param a: Vertical width of the localization window.
        :param b: Horizontal width of the localization window.  For example, if a=1 and b=2, then each window will be a
            3x5 rectangle centered at the active pixel.
        """
        self.m = m
        self.n = n
        self.dim = m * n
        # setup trivial image partition
        partition = TrivialImagePartition(m=m, n=n)
        # now, setup rectangular windows around each pixel
        window_list = self._setup_windows(a, b)
        # Also set up localization windows
        localization_window_list = self._setup_windows(a + c, b + d)
        # from each window, make the corresponding localization functional by weighting all pixels equally
        filter_list = []
        for i in range(self.m * self.n):
            localization_window_i = localization_window_list[i]
            window_i = window_list[i]
            # find the indices of window_i in localization_window_i
            window_indices = find_indices(subarr=window_i, arr=localization_window_i)
            # make the weight vector where all indices in window_i are weighted equally
            k = window_i.size
            weights_i = np.zeros(localization_window_i.size)
            weights_i[window_indices] = 1 / k
            lfunctional_i = LinearFilter(indices=localization_window_i, weights=weights_i)
            filter_list.append(lfunctional_i)
        ImageFilterFunction.__init__(self, image_partition=partition, filter_list=filter_list)

    def _setup_windows(self, a: int, b: int) -> List[np.ndarray]:
        """
        Creates a list of windows around each pixel.
        :returns: A list of numpy arrays of the same length as 'self.windows'.
        """
        windows = []
        for i in range(self.dim):
            # get x-y coordinates associated to the active index
            index_coords = indices_to_coords(m=self.m, n=self.n, indices=i)
            # determine the upper-left and lower-right corner of the window
            upper_left = index_coords[:, 0] - np.array([a, b])
            lower_right = index_coords[:, 0] + np.array([a, b])
            # get indices of the window using the coordinates for its upper left and lower right corner
            window_i = rectangle_indices(self.m, self.n, upper_left, lower_right)
            windows.append(window_i)
        return windows