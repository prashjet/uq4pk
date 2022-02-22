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
    weights = np.array([[.8, 0.2], [.2, 0.8]])
    filter_function = um.SimpleFilterFunction(weights=weights)
    discretization = um.TrivialAdaptiveDiscretization(dim=2)
    # compute kernel-localized credible intervals
    fci_obj = um.fci(alpha=alpha, model=prob.model, x_map=prob.x_map, ffunction=filter_function,
                  discretization=discretization, options={"use_ray": False})
    fcis = fci_obj.interval
    # Assert that the kernel functionals evaluated at the MAP estimate lie inside the credible intervals.
    y_map = filter_function.evaluate(prob.x_map)
    assert np.all((fcis[:, 0] <= y_map) & (y_map <= fcis[:, 1]))