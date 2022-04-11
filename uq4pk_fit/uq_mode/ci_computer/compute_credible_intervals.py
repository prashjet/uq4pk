
import numpy as np

from ..evaluation import AffineEvaluationMap
from ..filter import FilterFunction
from ..linear_model import LinearModel
from .ci_computer import CIComputer
from .credible_intervals import CredibleInterval


def compute_credible_intervals(alpha: float, model: LinearModel, x_map: np.ndarray, aemap: AffineEvaluationMap,
                               options: dict) -> CredibleInterval:
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
    credible_interval = ci_computer.compute_all()
    return credible_interval