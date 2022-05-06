
import numpy as np
import pytest

from uq4pk_fit.uq_mode.fci.adapted_filter_functional import AdaptedFilterFunctional
from uq4pk_fit.uq_mode.discretization import TrivialAdaptiveDiscretization, TrivialDiscretization
from uq4pk_fit.uq_mode.filter import IdentityFilterFunction


@pytest.fixture
def dummy_filter_map_functional():
    m = 10
    n = 10
    N = m * n
    discretization = TrivialDiscretization(dim=N)
    filter_function = IdentityFilterFunction(dim=N)
    test_filter = filter_function.filter(0)
    dummy_map = np.arange(N)
    test_functional = AdaptedFilterFunctional(filter=test_filter, x_map=dummy_map, discretization=discretization)
    return test_filter, dummy_map, test_functional


def test_z0_is_mapped_to_map(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    zero_z = test_functional.z0
    x_zero = test_functional.x(zero_z)
    assert np.isclose(x_zero, dummy_map).all()


def test_phi_returns_phidim(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    z0 = test_functional.z0
    phi0 = test_functional.phi(z0)


def test_x_works(dummy_filter_map_functional):
    test_filter, dummy_map, test_functional = dummy_filter_map_functional
    z = np.random.randn(test_functional.zdim)
    x_z = test_functional.x(z)
    assert np.isclose(x_z, z).all()


