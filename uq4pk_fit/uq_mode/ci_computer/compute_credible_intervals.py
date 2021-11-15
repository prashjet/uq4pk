
import numpy as np

from ..evaluation import AffineEvaluationMap
from ..linear_model import LinearModel
from .ci_computer import CIComputer


def compute_credible_intervals(alpha: float, model: LinearModel, x_map: np.ndarray, aemap: AffineEvaluationMap,
                               options: dict):
    """
    Computes generalized credible intervals with respect to an :py:class:`EvaluationMap`.

    :param alpha: The credibility parameter.
    :param model: The underlying statistical model.
    :param x_map: The MAP estimate.
    :param aemap: Defines what kind of credible intervals are to be computed.
    :param options: A dictionary with further options.
    """
    # Initialize CIComputer object
    ci_computer = CIComputer(alpha=alpha, model=model, x_map=x_map, aemap=aemap, options=options)
    # Compute credible intervals
    credible_intervals = ci_computer.compute_all()
    return credible_intervals