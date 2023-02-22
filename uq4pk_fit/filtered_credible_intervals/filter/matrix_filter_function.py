import numpy as np

from .filter_function import FilterFunction
from .linear_filter import LinearFilter


class MatrixFilterFunction(FilterFunction):
    """
    Very simple filter function. For the uncertain parameter x in R^d and a (m, d)-matrix A, the individual filters are
        phi_j(v) = A_j @ v,
    where A_j is the j-th row of A.
    """
    def __init__(self, mat: np.ndarray):
        assert mat.ndim == 2
        dim = mat.shape[1]
        # Create list of individual linear filters, each corresponding to a row in the matrix.
        filter_list = [LinearFilter(weights=row) for row in mat]
        # That's it. We can call the constructor of the parent class.
        FilterFunction.__init__(self, dim=dim, filter_list=filter_list)


