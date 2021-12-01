
import numpy as np
from uq4pk_fit.special_operators import NormalizedLaplacian


def test_normalized_laplacian():
    m = 10
    n = 10
    scales = tuple([1 * 1.6 ** k for k in range(4)])
    # Initialize normalized laplacian
    normalized_laplacian = NormalizedLaplacian(m=m, n=n, scales=scales)
    # Check that the dimension is correct.
    assert normalized_laplacian.dim == len(scales) * m * n