
from .filter_function import FilterFunction
from ..partition.image_partition import ImagePartition


class ImageFilterFunction(FilterFunction):
    """
    Special case of FilterFunction for 2-dimensional parameters (images).
    """
    def __init__(self, image_partition: ImagePartition, filter_list: list):
        """
        :param image_partition: The underlying partition into superpixels.
        :param filter_list: The associated list of filters. Must have the same length as ``image_partition``.
        """
        self.m = image_partition.m
        self.n = image_partition.n
        FilterFunction.__init__(self, partition=image_partition, filter_list=filter_list)
