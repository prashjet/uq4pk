
import numpy as np
from typing import List

from ..filter import FilterFunction
from ..linear_model import LinearModel

from .rml_sampler import RMLSampler
from .samples_to_ci import samples_to_ci


def fci_rml(alpha, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction, options: dict = None) \
        -> np.ndarray:
    """
    Computes filtered credible intervals using (a version of) randomized maximum likelihood.

    :param alpha: The credibility parameter. For example, alpha = 0.05 corresponds to 95%-credibility.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param ffunction: A filter function that determines the filtering.
    :param options: A dictionary with additional options.
            - "nsamples": Number of Monte Carlo samples. Default is 1000.
            - "reduction": The reduction factor for the regularization parameter. Default is 10.
            - "tol": The tolerance for the CGN solver.
            - "maxiter": The maximum number of iterations for the CGN solver.
    :return: Returns the filtered credible interval as an array of shape (n,2).
    """
    # default options
    if options is None:
        options = {}
    # initialize sampler
    sampler = RMLSampler(model=model, x_map=x_map, options=options)
    # produce samples
    nsamples = options.setdefault("nsamples", 1000)
    samples = sampler.sample(nsamples)
    # compute credible intervals from samples
    credible_intervals = samples_to_ci(alpha, samples, ffunction)
    # return credible intervals
    return credible_intervals