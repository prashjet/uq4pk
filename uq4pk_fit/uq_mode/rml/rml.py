
import numpy as np

from ..linear_model import LinearModel
from .rml_sampler import RMLSampler


def rml(n_samples: int, model: LinearModel, x_map: np.ndarray, sample_prior: bool = False,
        options: dict = None) \
        -> np.ndarray:
    """
    Computes posterior samples for a given linear model using the randomized maximum likelihood.

    :param n_samples: Number of samples.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param sample_prior: If True, the regularizing guess is sampled from the prior in each step. If False, the
        regularizing prior is fixed at the initial guess.
    :param options: A dictionary with additional options.
            - "reduction": The reduction factor for the regularization parameter. Default is 10.
            - "tol": The tolerance for the CGN solver.
            - "maxiter": The maximum number of iterations for the CGN solver.
    :return: An array of shape (n_samples, d), where d = model.dim. The rows correspond to the independent samples.
    """
    # default options
    if options is None:
        options = {}
    # initialize sampler
    sampler = RMLSampler(model=model, x_map=x_map, sample_prior=sample_prior, options=options)
    # return samples
    samples = sampler.sample(n_samples)
    return samples