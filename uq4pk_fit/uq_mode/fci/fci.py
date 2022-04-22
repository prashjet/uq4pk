"""
Contains function "lci".
"""

import numpy as np

from ..ci_computer import compute_credible_intervals
from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from ..discretization import AdaptiveDiscretization
from .filter_function_to_evaluation_map import filter_function_to_evaluation_map


RTOL = 0.01


class FCI:
    """
    Object that is returned by the function "fci".
    """
    def __init__(self, phi_lower_enlarged: np.ndarray, phi_upper_enlarged: np.ndarray, time_avg: float = -1):
        self.interval = np.column_stack([phi_lower_enlarged, phi_upper_enlarged])
        self.time_avg = time_avg


def fci(alpha: float, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction,
        discretization: AdaptiveDiscretization, options: dict = None) \
        -> FCI:
    """
    Computes filtered credible intervals using the Pereyra approximation.

    :param alpha: The credibility parameter. For example, alpha = 0.05 corresponds to 95%-credibility.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param ffunction: A filter function that determines the filtering.
    :param discretization: The underlying discretization
    :param weights: Of the same shape as x_map. If provided, the uncertainty quantification is not performed with
        respect to the parameter x, but instead with respect to the rescaled parameter u = weights * x (entry-wise
        multiplication).
    :param options: A dictionary with additional options.
            - "use_ray": If True, then the computation is parallelized with the Ray framework. Default is True.
            - "num_cpus": Number of CPUs used by Ray.
            - "solver": The optimization solver. "slsqp" for SLSQP, "ecos" for ECOS solver.
            - "detailed": If True, then the solver also outputs all local optimizers. Default is False.
    :returns: Object of type :py:class:`FCI`.
    """
    # Check the input.
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must satisfy 0 < alpha < 1.")
    assert x_map.shape == (model.dim,)
    assert ffunction.dim == model.dim
    assert discretization.dim == model.dim

    if options is None: options = {}
    # Get the optional weights
    weights = options.setdefault("weights", None)
    # Generate an affine evaluation map from the filter function
    affine_evaluation_map = filter_function_to_evaluation_map(ffunction, discretization, x_map, weights=weights)
    # If subsample is not None, kick out all pixels that are not in sample.
    sample = options.setdefault("sample", None)
    if sample is not None:
        affine_evaluation_map.select(sample)
    # Compute the credible intervals (in phi-space)
    phi_map = ffunction.evaluate(x_map)
    scale = phi_map.max()
    credible_interval = compute_credible_intervals(alpha=alpha, model=model, x_map=x_map, aemap=affine_evaluation_map,
                                                   scale=scale, options=options)
    # Create FCI-object
    phi_lower = credible_interval.phi_lower
    phi_upper = credible_interval.phi_upper

    fci_obj = FCI(phi_lower_enlarged=phi_lower, phi_upper_enlarged=phi_upper, time_avg=credible_interval.time_avg)

    return fci_obj
