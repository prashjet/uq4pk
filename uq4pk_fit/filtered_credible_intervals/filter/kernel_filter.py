
import numpy as np
from typing import Literal, Union

from ..geometry2d import indices_to_coords
from .linear_filter import LinearFilter
from .filter_kernel import FilterKernel


class KernelFilter(LinearFilter):
    """
    A distance filter is a filter that is determined by the distance function. The distance function is then
    translated into a weight via a weighting function.
    """
    def __init__(self, m: int, n: int, center: np.ndarray, kernel: FilterKernel,
                 boundary: Literal["reflect", "zero"] = "zero"):
        # Define the weights
        dim = m * n
        all_indices = np.arange(dim)
        all_coords = indices_to_coords(m=m, n=n, indices=all_indices)
        all_distances = all_coords - center[:, np.newaxis]
        w = kernel.weighting(all_distances)
        # By normalizing the weights, we can implicitly define the type of boundary.
        if boundary == "reflect":
            w = w / np.sum(w)
        elif boundary == "zero":
            w = w
        else:
            raise ValueError("'boundary' has to be 'zero' or 'reflect'.")
        LinearFilter.__init__(self, weights=w)