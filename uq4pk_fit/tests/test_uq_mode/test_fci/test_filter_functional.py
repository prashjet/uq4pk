
import numpy as np
import pytest

from uq4pk_fit.uq_mode.fci.filter_functional import FilterFunctional
from uq4pk_fit.uq_mode.filter import ExponentialFilterFunction


@pytest.fixture
def dummy_filter_map_functional():
    m = 10
    n = 10
    N = m * n
    filter_function = ExponentialFilterFunction(m=m, n=n, a=2, b=2, c=2, d=2)
    test_filter = filter_function.filter(0)
    dummy_map = np.arange(N)
    test_functional = FilterFunctional(test_filter, x_map=dummy_map)
    return test_filter, dummy_map, test_functional


def test_the_filter_parameters_are_copied_correctly(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    assert test_functional.dim == dummy_map.size
    assert np.isclose(test_functional.w, test_filter.weights).all()
    assert np.isclose(test_functional.indices, test_filter.indices).all()
    assert test_functional.zdim == test_filter.indices.size


def test_z0_is_mapped_to_map(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    zero_z = test_functional.z0
    x_zero = test_functional.x(zero_z)
    assert np.isclose(x_zero, dummy_map).all()


def test_phi_returns_phidim(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    z0 = test_functional.z0
    phi0 = test_functional.phi(z0)
    assert phi0.size == test_functional.phidim


def test_x_works(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    z = np.random.randn(test_functional.zdim)
    x_z = test_functional.x(z)
    assert np.isclose(x_z[test_functional.indices], z).all()
    inactive_indices = np.delete(np.arange(test_functional.dim), test_functional.indices)
    assert np.isclose(x_z[inactive_indices], dummy_map[inactive_indices]).all()


