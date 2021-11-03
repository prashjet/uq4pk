"""
Test for the function "lci".
"""

import numpy as np

import uq4pk_fit.tests.test_uq_mode.unconstrained_problem as testproblem
import uq4pk_fit.uq_mode as uq_mode

def test_lci():
    # get the test problem
    prob = testproblem.get_problem()

    # compute Cui's locally credible intervals
    alpha = 0.05
    # make partition consisting of one element
    partition = uq_mode.partition.Partition(dim=2, elements=[np.array([0, 1])])
    # compute localized credible intervals
    lci = uq_mode.lci(alpha=alpha, model=prob.model, x_map=prob.x_map, partition=partition)
    # Assert that lower bound is less than upper bound
    assert np.all((lci[:, 0] <= lci[:, 1]))


test_lci()