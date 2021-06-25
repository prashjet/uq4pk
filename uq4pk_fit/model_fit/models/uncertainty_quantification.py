"""
Contains function "uq"
"""

from uq4pk_fit.uq_mode import *

from .model import FittedModel


def uq(fitted_model: FittedModel, alpha, partition):
    """
    Computes local credible intervals for a fitted model
    :param fitted_model: FittedModel
    :param alpha: float (0 < alpha < 1)
    :return: ndarray (shape=(fitted_model.dim,2))
    """
    intervals = rci(alpha=alpha,
                    partition=partition,
                    n=fitted_model.dim,
                    xmap=fitted_model.map,
                    cost=fitted_model.costfun,
                    lb=fitted_model.lb)
    return intervals