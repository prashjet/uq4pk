
import numpy as np
from .image_filter_function import ImageFilterFunction
from .linearfilter import LinearFilter
from ..partition import rectangle_partition


def upsample_ffunction(ffunction: ImageFilterFunction, m: int, n: int, a: int, b: int):
    """
    From an ImageFilterFunction defined on the superpixels, creates the corresponding ImageFilterFunction defined on
    the original image.

    :param ffunction: A filter function for the downsampled image.
    :param m: The number of rows for the original image.
    :param n: The number of columns for the original image.
    :param a: The pixel-height of the superpixels.
    :param b: The pixel-width of the superpixels.
    """
    upsampler = Upsampler(m, n, a, b)
    upsampled_ffunction = upsampler.upsample_ffunction(ffunction)
    return upsampled_ffunction


class Upsampler:
    """
    Manages the downsampling of an image.
    We are given an image of shape (m, n) and parameters a, b that determine the height and width of the superpixels.
    The downsampling object then treats then handles how this image is translated to an image of shape
    (ceil(m/a), ceil(n/b)).
    """
    def __init__(self, m: int, n: int, a: int, b: int):
        """
        :param m: Number of rows of the original image.
        :param n: Number of columns of the original image.
        :param a: Number of rows in each superpixel.
        :param b: Number of columns in each superpixel.

        As an example, parameter values (m, n, a, b) = (10, 30, 2, 3) mean that the original image has shape 10x30
        pixels and is partitioned into superpixels of 2x3 pixels.
        """
        # make rectangle partition
        self._superpixels = rectangle_partition(m, n, a, b)
        # get shape of downsampled image
        self._m_down = np.floor(m / a).astype(int)
        self._n_down = np.floor(n / b).astype(int)

    def upsample_ffunction(self, ffunction: ImageFilterFunction):
        """
        Translates a filter function for the downsampled image to a filter function for the upsampled image.
        :param ffunction:
        :returns: An image filter function for the original image.
        """
        # check that ffunction is indeed defined for the downsampled image
        assert ffunction.m == self._m_down
        assert ffunction.n == self._n_down
        # check that ffunction has indeed the right number of entries
        assert ffunction.size == self._superpixels.size
        # initialize filter list
        upsampled_filter_list = []
        for i in range(ffunction.size):
            filter_i = ffunction.filter(i)
            indices_list = []
            weights_list = []
            for j in range(filter_i.size):
                indices_j = self.upsampling_map(filter_i.indices[j])
                weight_j = (filter_i.weights[j] / indices_j.size) * np.ones(indices_j.size)
                indices_list.append(indices_j)
                weights_list.append(weight_j)
            # concatenate indices and weights-vectors
            indices_i = np.concatenate(indices_list)
            weights_i = np.concatenate(weights_list)
            assert indices_i.size == weights_i.size
            upsampled_filter_i = LinearFilter(indices=indices_i, weights=weights_i)
            upsampled_filter_list.append(upsampled_filter_i)
        # make the filter function and return it
        upsampled_filter_function = ImageFilterFunction(image_partition=self._superpixels,
                                                        filter_list=upsampled_filter_list)
        return upsampled_filter_function

    def upsampling_map(self, i: int) -> np.ndarray:
        """
        Returns the indices of all pixels inside a given superpixel.

        :param i: Index of the superpixel in the underlying partition.
        :return: Indices of all pixels as array of type int.
        """
        superpixel = self._superpixels.element(i)
        return superpixel