"""
Test for rml.RMLSampler.
"""

import numpy as np

import cgn2 as cgn

import tests.unconstrained_problem as testproblem

from uq_mode.rml.rml_sampler import RMLSampler



def test_rml_sampler():
    # Get test_problem
    test_problem = testproblem.get_problem()
    # Setup model
    xbar = test_problem.xbar
    regop = cgn.IdentityOperator(dim=xbar.size)
    delta = test_problem.delta
    noise_regop = cgn.DiagonalOperator(dim=2, s=1/delta)

    def forward(x):
        return noise_regop.fwd(test_problem.H @ x - test_problem.y)

    def forward_jac(x):
        return noise_regop.fwd(test_problem.H)

    # Setup MultiParameterProblem
    pars = cgn.Parameters()
    pars.add_parameter(dim=2, mean=xbar, p=regop)
    model = cgn.MultiParameterProblem(fun=forward, jac=forward_jac, params=pars)
    # Initialize sampler from test_problem.
    sampler = RMLSampler(model=model)
    # Create samples
    nsamples = 3000
    samples = sampler.sample(nsamples)
    # Check that the sample-array has the right shape
    assert samples.shape == (xbar.size, nsamples)
    # The average of the samples should be close to x_map
    sample_mean = np.mean(samples, axis=1)
    xmap = test_problem.xmap
    err_mean = np.linalg.norm(sample_mean - xmap) / np.linalg.norm(xmap)
    print(f"err mean = {err_mean}")
    assert err_mean < 0.1
    # The sample covariance should be close to the posterior covariance.
    sample_cov = np.cov(samples)
    cov_post = test_problem.s_post @ test_problem.s_post.T
    err_cov = np.linalg.norm(sample_cov - cov_post) / np.linalg.norm(cov_post)
    print(f"err_cov: {err_cov}")
    assert err_cov < 0.1
    x1 = np.random.randn(nsamples)
    x2 = np.random.randn(nsamples)
    x_s = np.row_stack([x1, x2])
    should_be_identity = np.cov(x_s)
    err_cov = np.linalg.norm(should_be_identity - np.identity(2))
    print(f"err_cov: {err_cov}")




test_rml_sampler()