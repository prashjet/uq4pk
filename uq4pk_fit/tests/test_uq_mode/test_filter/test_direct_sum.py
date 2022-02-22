
import numpy as np

from uq4pk_fit.uq_mode.filter import direct_sum, GaussianFilterFunction2D, IdentityFilterFunction


def test_direct_sum():
    m = 12
    n = 53
    d1 = m * n
    d2 = 7
    ffunction1 = GaussianFilterFunction2D(m=m, n=n, sigma=1., boundary="zero")
    ffunction2 = IdentityFilterFunction(dim=d2)
    ffunction = direct_sum(ffunction1, ffunction2)
    assert ffunction.dim == d1 + d2
    # first filter from second ffunction
    s1 = ffunction1.dim
    test_filter = ffunction.filter(s1)
    assert np.isclose(test_filter.weights[d1:], ffunction2.filter(0).weights).all()