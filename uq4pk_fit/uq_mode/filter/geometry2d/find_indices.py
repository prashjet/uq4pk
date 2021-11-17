
import numpy as np


def find_indices(arr: np.ndarray, subarr: np.ndarray) -> np.ndarray:
    """
    Finds indices of subarray in larger array.
    :param subarr:
    :param arr:
    """
    # Check that subarr is indeed subarr
    assert set(subarr).issubset(arr)
    # Find indices of subarr in arr.
    sorter = np.argsort(arr)
    indices = sorter[np.searchsorted(arr, subarr, sorter=sorter)]
    return indices