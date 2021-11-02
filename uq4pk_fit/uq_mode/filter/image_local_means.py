"""
Contains class "ImageLocalMeans".
"""

from .image_filter_function import ImageFilterFunction
from .geometry2d import windowed_rectangles
from .local_mean_filter import LocalMeanFilter


class ImageLocalMeans(ImageFilterFunction):
    """
    A filter function that associates to each superpixel the corresponding local means with a localization frame of
    prescribed width.
    """
    def __init__(self, m, n, a, b, c, d):
        """
        :param m: int
            Number of rows of the image.
        :param n: int
            Number of columns of the image.
        :param a: int
            Number of rows of each superpixel.
        :param b: int
            Number of columns of each superpixel.
        :param c: int
            Vertical width of the localization frame.
        :param d: int
            Horizontal width of the localization frame.
        """
        partition, window_list = windowed_rectangles(m, n, a, b, c, d)
        superpixel_list = partition.get_element_list()
        filter_list = []
        for superpixel, window in zip(superpixel_list, window_list):
            filter = LocalMeanFilter(indices=superpixel, window=window)
            filter_list.append(filter)
        ImageFilterFunction.__init__(self, image_partition=partition, filter_list=filter_list)