
from .image_filter_function import ImageFilterFunction
from .geometry2d import windowed_rectangles
from .local_mean_filter import LocalMeanFilter


class ImageLocalMeans(ImageFilterFunction):
    """
    A filter function that associates to each superpixel the corresponding local means with a localization frame of
    prescribed width.
    """
    def __init__(self, m: int, n: int, a: int, b: int, c: int, d: int):
        """
        :param m: Number of rows of the image.
        :param n: Number of columns of the image.
        :param a: Number of rows of each superpixel.
        :param b: Number of columns of each superpixel.
        :param c: Vertical width of the localization frame.
        :param d: Horizontal width of the localization frame.
        """
        # Create the underlying partition.
        partition, window_list = windowed_rectangles(m, n, a, b, c, d)
        superpixel_list = partition.get_element_list()
        # Create the associated list of LinearFilter objects.
        filter_list = []
        for superpixel, window in zip(superpixel_list, window_list):
            filter = LocalMeanFilter(indices=superpixel, window=window)
            filter_list.append(filter)
        ImageFilterFunction.__init__(self, image_partition=partition, filter_list=filter_list)