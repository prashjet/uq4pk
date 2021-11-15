
import numpy as np

from uq4pk_fit.uq_mode.filter import direct_sum, ExponentialFilterFunction, IdentityFilterFunction


def test_direct_sum():
    m = 12
    n = 53
    d1 = m * n
    d2 = 7
    ffunction1 = ExponentialFilterFunction(m=m, n=n, a=1, b=1, c=2, d=2)
    ffunction2 = IdentityFilterFunction(dim=d2)
    ffunction = direct_sum([ffunction1, ffunction2])
    assert ffunction.dim == d1 + d2
    assert ffunction.size == ffunction1.size + ffunction2.size
    # first filter from second ffunction
    s1 = ffunction1.size
    indices_should_be = np.arange(d1, d1 + d2)
    test_filter = ffunction.filter(s1)
    assert np.all(test_filter.indices == indices_should_be)
    assert np.isclose(test_filter.weights, ffunction2.filter(0).weights).all()