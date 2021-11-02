"""
Contains class "TrivialImagePartition".
"""

import numpy as np

from .image_partition import ImagePartition


class TrivialImagePartition(ImagePartition):

    def __init__(self, m, n):
        dim = m * n
        element_list = []
        for i in range(dim):
            element_list.append(np.array([i]))
        ImagePartition.__init__(self, m=m, n=n, superpixel_list=element_list)