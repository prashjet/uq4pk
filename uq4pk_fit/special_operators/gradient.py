
import numpy as np
from typing import List


def mygradient(arr: np.ndarray) -> List[np.ndarray]:
    """
    Computes the gradient of an N-dimensional array, with reflection at the edges.

    :param arr: The array for which the gradient is to be computed.
    :returns: A list of the gradient fields in each direction. Each gradient field has the same shape as ``arr``.
        For example, for an array of shape (m, n), ``gradient`` will return two arrays of shape (m, n), corresponding
        to the gradients in x- and y-direction.
    """
    N = arr.ndim
    axes = tuple(range(N))
    # Pad array by nearest value
    padded_arr = np.pad(arr, pad_width=1, mode="reflect")
    # Make slice that removes all boundary values
    slc = [slice(1,-1)] * N
    gradients = []
    for axis in axes:
        arr_plus = np.roll(a=padded_arr, shift=-1, axis=axis)
        grad = arr_plus - padded_arr
        # remove padded values
        grad = grad[slc]
        # Check that dimensions match
        assert grad.shape == arr.shape
        gradients.append(grad)
    # Check that we have the right number of gradients.
    assert len(gradients) == N
    return gradients

