
import numpy as np

from uq4pk_fit.uq_mode.filter import ExponentialFilterFunction
from uq4pk_fit.uq_mode.fci.filter_function_to_evaluation_map import filter_function_to_evaluation_map


def test_filter_to_evaluation_map():
    # as a test function, we use exponential
    n1 = 12
    n2 = 50
    x_map = np.ones((n1 * n2, ))
    ffunction = ExponentialFilterFunction(m=n1, n=n2, a=2, b=2, c=4, d=4)
    evaluation_map = filter_function_to_evaluation_map(ffunction, x_map)
    assert evaluation_map.size == ffunction.size
    assert evaluation_map.dim == ffunction.dim
    # finally, compare individual filters
    for aef, filter in zip(evaluation_map.aef_list, ffunction.get_filter_list()):
        assert np.isclose(aef.w, filter.weights).all()
        z_test = np.arange(aef.zdim)
        xtest1 = aef.x(z_test)
        xtest2 = x_map.copy()
        xtest2[filter.indices] = z_test
        assert np.isclose(xtest1, xtest2).all()
