
import numpy as np

from uq4pk_fit.uq_mode.filter.geometry2d import find_indices


def test_find_indices():
    arr = np.array([0, 1, 10, 7, 3, 2, 4, 42])
    subarr = np.array([0, 1, 42, 3])
    indices = find_indices(arr=arr, subarr=subarr)
    assert np.isclose(indices, np.array([0, 1, 7, 4])).all()
