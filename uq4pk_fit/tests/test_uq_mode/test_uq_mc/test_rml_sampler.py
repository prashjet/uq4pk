"""
Test for rml.RMLSampler.
"""

import numpy as np

import uq4pk_fit.cgn as cgn

import uq4pk_fit.tests.test_uq_mode.unconstrained_problem as testproblem

from uq4pk_fit.uq_mode.uq_mc.rml_sampler import RMLSampler



def test_rml_sampler():
    # Get test_problem
    test_problem = testproblem.get_problem()
    # Initialize sampler from test_problem.
    sampler = RMLSampler(model=test_problem.model, x_map=test_problem.x_map, options={})
    # Create samples
    nsamples = 3000
    samples = sampler.sample(nsamples)
    # Check that the sample-array has the right shape
    assert samples.shape == (test_problem.x_map.size, nsamples)
    # The average of the samples should be close to x_map
    sample_mean = np.mean(samples, axis=1)
    xmap = test_problem.x_map
    err_mean = np.linalg.norm(sample_mean - xmap) / np.linalg.norm(xmap)
    print(f"err mean = {err_mean}")
    assert err_mean < 0.1




test_rml_sampler()