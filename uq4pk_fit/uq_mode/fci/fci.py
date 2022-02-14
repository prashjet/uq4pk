"""
Contains function "lci".
"""

import numpy as np
from typing import Sequence

from ..ci_computer import compute_credible_intervals
from ..filter.filter_function import FilterFunction
from ..linear_model import LinearModel
from .filter_function_to_evaluation_map import filter_function_to_evaluation_map


class FCI:
    """
    Object that is returned by the function "fci".
    """
    def __init__(self, phi_lower_enlarged: np.ndarray, phi_upper_enlarged: np.ndarray, minimizers: Sequence[np.ndarray],
                 maximizers: Sequence[np.ndarray]):
        self.interval = np.column_stack([phi_lower_enlarged, phi_upper_enlarged])
        self.upper = phi_upper_enlarged
        self.minimizers = minimizers
        self.maximizers = maximizers


def fci(alpha: float, model: LinearModel, x_map: np.ndarray, ffunction: FilterFunction, options: dict = None) \
        -> FCI:
    """
    Computes filtered credible intervals using the Pereyra approximation.

    :param alpha: The credibility parameter. For example, alpha = 0.05 corresponds to 95%-credibility.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param ffunction: A filter function that determines the filtering.
    :param options: A dictionary with additional options.
            - "use_ray": If True, then the computation is parallelized with the Ray framework. Default is True.
            - "num_cpus": Number of CPUs used by Ray.
            - "solver": The optimization solver. "slsqp" for SLSQP, "ecos" for ECOS solver.
            - "detailed": If True, then the solver also outputs all local optimizers. Default is False.
            - "tilde": If True, then the credible intervals are computed with respect to the smaller treshold
                :math:`\\tilde \\gamma_\\alpha = \\hat \\gamma_\\alpha - \\eta_\\alpha \\sqrt{n} + n`.
    :returns: Object of type :py:class:`FCI`.
    """
    _check_input(alpha, model, x_map, ffunction)
    # Generate an affine evaluation map from the filter function
    affine_evaluation_map = filter_function_to_evaluation_map(ffunction, x_map)
    # If subsample is not None, kick out all pixels that are not in sample.
    sample = options["sample"]
    if sample is not None:
        affine_evaluation_map.select(sample)
    # Compute the credible intervals (in phi-space)
    credible_interval = compute_credible_intervals(alpha=alpha, model=model, x_map=x_map, aemap=affine_evaluation_map,
                                                    options=options)
    # Create FCI-object
    phi_lower = credible_interval.phi_lower
    phi_upper = credible_interval.phi_upper
    if sample is None:
        phi_lower_enlarged = ffunction.enlarge(phi_lower)
        phi_upper_enlarged = ffunction.enlarge(phi_upper)
    else:
        phi_lower_enlarged = phi_lower
        phi_upper_enlarged = phi_upper
    fci_obj = FCI(phi_lower_enlarged=phi_lower_enlarged, phi_upper_enlarged=phi_upper_enlarged,
                  minimizers=credible_interval.minimizers, maximizers=credible_interval.maximizers)

    return fci_obj


def _check_input(alpha, model, x_map, ffunction):
    """
    Checks the input of "fci" for consistency.
    """
    if not 0 < alpha < 1:
        raise ValueError("'alpha' must satisfy 0 < alpha < 1.")
    assert x_map.shape == (model.n, )
    assert ffunction.dim == model.n
