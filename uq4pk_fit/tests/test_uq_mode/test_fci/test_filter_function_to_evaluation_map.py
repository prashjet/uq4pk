
import numpy as np

from uq4pk_fit.uq_mode.filter import GaussianFilterFunction2D
from uq4pk_fit.uq_mode.discretization import TrivialAdaptiveDiscretization
from uq4pk_fit.uq_mode.fci.filter_and_discretization_to_evaluation_map import filter_and_discretization_to_evaluation_map


def test_filter_to_evaluation_map():
    # as a test function, we use exponential
    n1 = 12
    n2 = 50
    x_map = np.ones((n1 * n2, ))
    ffunction = GaussianFilterFunction2D(m=n1, n=n2, sigma=1., boundary="zero")
    discretization = TrivialAdaptiveDiscretization(dim=n1 * n2)
    evaluation_map = filter_and_discretization_to_evaluation_map(filter_function=ffunction,
                                                                 discretization=discretization, x_map=x_map)

    assert evaluation_map.dim == ffunction.dim
    # finally, compare individual filters
    for aef, filter in zip(evaluation_map.aef_list, ffunction.get_filter_list()):
        z_test = np.arange(aef.zdim)
        xtest1 = aef.x(z_test)
        xtest2 = z_test
        assert np.isclose(xtest1, xtest2).all()
