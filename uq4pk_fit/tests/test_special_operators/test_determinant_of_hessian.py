
import numpy as np

from uq4pk_fit.special_operators import DeterminantOfHessian
from uq4pk_fit.special_operators.doh import doh


def test_determinant_of_hessian():
    m = 12
    n = 53

    testim = np.random.randn(m, n)
    # Evaluate DOH-operator on image.
    doh_op = DeterminantOfHessian(testim.shape).mat
    doh_1 = doh_op @ testim.flatten()
    doh_1 = np.reshape(doh_1, testim.shape)

    # For comparison, compute DOH of test image directly.
    doh_2 = doh(testim)

    # They should be the same.
    assert np.isclose(doh_1, doh_2).all()