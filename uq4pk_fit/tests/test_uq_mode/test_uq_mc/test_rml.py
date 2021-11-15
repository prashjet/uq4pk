"""
Test for "fci.py"
"""


import numpy as np

import uq4pk_fit.cgn as cgn
import uq4pk_fit.uq_mode as uq_mode

import uq4pk_fit.tests.test_uq_mode.unconstrained_problem as testproblem


def test_rml():
    # get the test problem
    test_problem = testproblem.get_problem()
    # compute filtered credible intervals
    alpha = 0.05
    # make a simple filter function
    weights = [np.array([.8, 0.2]), np.array([.2, 0.8])]
    filter_function = uq_mode.SimpleFilterFunction(dim=2, weights=weights)
    rmlci = uq_mode.fci_rml(alpha=alpha, model=test_problem.model, x_map=test_problem.x_map, ffunction=filter_function)
    # Assert that the kernel functionals evaluated at the MAP estimate lie inside the credible intervals.
    y_map = filter_function.evaluate(test_problem.x_map)
    assert np.all((rmlci[:, 0] <= y_map) & (y_map <= rmlci[:, 1]))