import numpy as np
import pytest

from uq4pk_fit.uq_mode.optimization import SOCP


@pytest.fixture
def socp_fixture():
    n = 23
    x_start = np.ones(n) / n
    w = np.arange(n)
    a = np.ones((1, n))
    b = np.ones((1, ))
    c = np.diag(np.arange(1, n+1))
    d = c @ x_start + 1.
    e = np.sum(np.square(c @ x_start - d)) + 0.5
    lb = np.zeros(n)
    # make sure that starting point satisfies all constraints
    assert np.isclose(a @ x_start - b, 0).all()
    assert e - np.sum(np.square(c @ x_start - d)) >= 0
    assert np.all(x_start >= lb)
    socp = SOCP(w=w, a=a, b=b, c=c, d=d, e=e, lb=lb, minmax=0)
    return [socp, x_start]
