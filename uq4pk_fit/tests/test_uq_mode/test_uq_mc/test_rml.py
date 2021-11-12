"""
Test for "fci.py"
"""


import numpy as np

from uq_mode.external_packages import cgn

import tests.unconstrained_problem as testproblem
import uq_mode


def test_rml():
    # get the test problem
    test_problem = testproblem.get_problem()
    # compute filtered credible intervals
    alpha = 0.05
    # make a simple filter function
    weights = [np.array([.8, 0.2]), np.array([.2, 0.8])]
    filter_function = uq_mode.SimpleFilterFunction(dim=2, weights=weights)
    # Setup the model
    xbar = test_problem.xbar
    regop = cgn.IdentityOperator(dim=xbar.size)
    delta = test_problem.delta
    noise_regop = cgn.DiagonalOperator(dim=2, s=1 / delta)
    def forward(x):
        return test_problem.H @ x

    def forward_jac(x):
        return test_problem.H
    # Setup model
    model = uq_mode.rml.Model(mean_list=[xbar], regop_list=[regop], forward=forward, forward_jac=forward_jac,
                  regop_noise=noise_regop)
    rmlci = uq_mode.rml.rml_ci(alpha=alpha, ffunction=filter_function, model=model, y=test_problem.y)
    # Assert that the kernel functionals evaluated at the MAP estimate lie inside the credible intervals.
    y_map = filter_function.evaluate(test_problem.xmap)
    assert np.all((rmlci[:, 0] <= y_map) & (y_map <= rmlci[:, 1]))


test_rml()