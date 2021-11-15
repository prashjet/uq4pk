"""
LinearFilter function associated to "DistanceFilter".
"""

import numpy as np

from .distance_filter import DistanceFilter
from .downsampling import upsample_ffunction
from .geometry2d import indices_to_coords
from .image_filter_function import ImageFilterFunction
from ..partition import TrivialImagePartition


class DistanceFilterFunction(ImageFilterFunction):
    """
    Special case of ImageFilterFunction that associates to each pixel a correspondingly positioned distance filter.
    See also :py:class:`DistanceFilter`.
    """
    def __init__(self, m: int, n: int, a: int, b: int, c: int, d: int, h: float):
        """
        :param m: Number of rows of the image.
        :param n: Number of columns of the image.
        :param a: Number of rows of each superpixel (for downsampling).
        :param b: Number of columns of each superpixel (for downsampling also).
        :param c: Vertical width of the window (in superpixels!)
        :param d: Horizontal width of the window (in superpixels!)
        :param h: Scaling factor. The distance will by divided by h before it is forwarded to the weighting functino.
        """
        # Make a filter for the downsampled image
        m_down = np.floor(m / a).astype(int)
        n_down = np.floor(n / b).astype(int)
        c_down = np.floor(c / a).astype(int)
        d_down = np.floor(d / b).astype(int)
        trivial_partition = TrivialImagePartition(m=m_down, n=n_down)
        filter_list = []
        index_list = trivial_partition.get_element_list()
        for index in index_list:
            # Get x-y-coordinates of "index" in the (m,n)-image.
            position = indices_to_coords(m_down, n_down, index).flatten()
            filter = DistanceFilter(m=m_down, n=n_down, a=c_down, b=d_down, position=position, weighting=self._weighting,
                                    scaling = h / np.array([a, b]))
            filter_list.append(filter)
        downsampled_ffunction = ImageFilterFunction(image_partition=trivial_partition, filter_list=filter_list)
        # translate the filter function for the superpixel-image to a filter function for the original image
        upsampled_ffunction = upsample_ffunction(downsampled_ffunction, m=m, n=n, a=a, b=b)
        # Initialize self by copying the upsampled ffunction
        ImageFilterFunction.__init__(self, image_partition=upsampled_ffunction._partition,
                                     filter_list=upsampled_ffunction.get_filter_list())

    # TO BE IMPLEMENTED

    def _weighting(self, d):
        """
        Weighting function for the filter. To each pixel, a weight is associated equal to weighting(d), where d is the
        distance to the center pixel.
        See also :py:class:`DistanceFilter`.
        """
        raise NotImplementedError