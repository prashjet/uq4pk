
import numpy as np

from ..evaluation import AffineEvaluationMap
from ..linear_model import LinearModel
from .ci_computer import CIComputer


def compute_credible_intervals(alpha: float, model: LinearModel, x_map: np.ndarray, aemap: AffineEvaluationMap,
                               options: dict):
    # Initialize CIComputer object
    ci_computer = CIComputer(alpha=alpha, model=model, x_map=x_map, aemap=aemap, options=options)
    # Compute credible intervals
    credible_intervals = ci_computer.compute()
    return credible_intervals