"""
Computes filtered credible intervals using a randomized maximum likelihood approach.
"""

from typing import List

from ..external_packages import cgn

from ..filter import FilterFunction

from .rml_sampler import RMLSampler
from .samples_to_ci import samples_to_ci


def rml_ci(alpha, problem: cgn.Problem, starting_values: List, ffunction: FilterFunction, nsamples=100,
           reduction: float=1., solver_options: dict=None, return_samples=False):
    """
    :param alpha: float
        The credibility parameter. Must satisfy 0 < alpha < 1.
        For example, if alpha = 0.05, then rml_ci returns the 95%-posterior credible intervals.
    :param problem: cgn.problem
        The statistical model.
    :param ffunction: FilterFunction
        A (concatenated) fci.FilterFunction object that determines how the credible intervals are computed from the
        samples.
    :param nsamples: int, optional
        Determines the number of samples to be generated. Defaults to 100.
    :return: array_like, (N,2)
        Returns the filtered credible interval as an array, where N is the dimension of the parameter space.
    """
    # initialize sampler
    sampler = RMLSampler(problem=problem, starting_values=starting_values, solver_options=solver_options,
                         reduction=reduction)
    # produce samples
    samples = sampler.sample(nsamples)
    # compute credible intervals from samples
    credible_intervals = samples_to_ci(alpha, samples, ffunction)
    # return credible intervals
    if return_samples:
        return credible_intervals, samples
    else:
        return credible_intervals