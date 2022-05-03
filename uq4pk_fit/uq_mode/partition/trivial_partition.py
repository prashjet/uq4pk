"""
Contains class "TrivialPartition".
"""

import numpy as np

from .partition import Partition


class TrivialPartition(Partition):

    def __init__(self, dim):
        """
        :param dim: int
        """
        element_list = []
        for i in range(dim):
            element_list.append(np.array([i]))
        Partition.__init__(self, dim=dim, elements=element_list)