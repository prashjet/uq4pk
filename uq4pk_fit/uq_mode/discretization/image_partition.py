"""
Contains class "ImagePartition".
"""

from .partition import Partition

class ImagePartition(Partition):

    def __init__(self, m, n, superpixel_list):
        self.m = m
        self.n = n
        dim = m * n
        Partition.__init__(self, dim=dim, elements=superpixel_list)