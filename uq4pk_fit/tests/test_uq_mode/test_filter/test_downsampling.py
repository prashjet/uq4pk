import numpy as np

from uq4pk_fit.uq_mode.filter.downsampling import upsample_ffunction
from uq4pk_fit.uq_mode.filter import ExponentialFilterFunction


def test_upsample_ffunction():
    m = 10
    n = 30
    c = 3
    d = 3
    a = 2
    b = 3
    m_down = int(m / a)
    n_down = int(n / b)
    c_down = np.floor(c / a).astype(int)
    d_down = np.floor(d / a).astype(int)
    downsampled_ffunction = ExponentialFilterFunction(m=m_down, n=n_down, a=1, b=1, c=c_down, d=d_down)
    ffunction = upsample_ffunction(downsampled_ffunction, m=m, n=n, a=a, b=b)
    assert ffunction.m == m
    assert ffunction.n == n
    assert ffunction.size == downsampled_ffunction.size
    assert ffunction.dim == m * n
