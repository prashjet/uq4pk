
import numpy as np
from pathlib import Path

from uq4pk_fit.blob_detection.blankets.compute_blanket import compute_blanket
from .parameters import EXAMPLE_MAP, EXAMPLE_LOWER, EXAMPLE_UPPER, SLICE_INDEX


loc = Path(__file__).parent


def one_dimensional_example():
    """
    Sets up the arrays for creating the plot that illustrated t-blankets in one dimension.
    """
    lower2d = np.load(str(loc / EXAMPLE_LOWER))
    upper2d = np.load(str(loc / EXAMPLE_UPPER))
    map2d = np.load(str(loc / EXAMPLE_MAP))
    n = lower2d.shape[1]
    lower = lower2d[SLICE_INDEX, :].reshape((n,))
    upper = upper2d[SLICE_INDEX, :].reshape((n,))
    map1d = map2d[SLICE_INDEX, :].reshape((n,))
    string = compute_blanket(lb=lower, ub=upper)
    return lower, upper, map1d, string