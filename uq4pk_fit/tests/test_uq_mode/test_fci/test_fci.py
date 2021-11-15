"""
Test for "fci.py"
"""


import numpy as np

import uq4pk_fit.tests.test_uq_mode.unconstrained_problem as testproblem
import uq4pk_fit.uq_mode as um


def test_fci():
    # get the test problem
    prob = testproblem.get_problem()
    # compute filtered credible intervals
    alpha = 0.05
    # make a simple filter function
    weights = [np.array([.8, 0.2]), np.array([.2, 0.8])]
    filter_function = um.SimpleFilterFunction(dim=2, weights=weights)
    # compute kernel-localized credible intervals
    fcis = um.fci(alpha=alpha, model=prob.model, x_map=prob.x_map, ffunction=filter_function,
                  options={"use_ray": False})
    # Assert that the kernel functionals evaluated at the MAP estimate lie inside the credible intervals.
    y_map = filter_function.evaluate(prob.x_map)
    assert np.all((fcis[:, 0] <= y_map) & (y_map <= fcis[:, 1]))