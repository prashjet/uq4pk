
from typing import List

from .linear_filter import LinearFilter
from .filter_function import FilterFunction


class ImageFilterFunction(FilterFunction):
    """
    Special case of FilterFunction for 2-dimensional parameters (images).
    """
    def __init__(self, m: int, n: int, filter_list: List[LinearFilter]):
        dim = m * n
        FilterFunction.__init__(self, dim=dim, filter_list=filter_list)
        self.m = m
        self.n = n
