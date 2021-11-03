"""
Contains the function "lci".
"""

import numpy as np

from .lci_computer import LCIComputer
from ..linear_model import LinearModel
from ..partition import Partition


def lci(alpha: float,  model: LinearModel, x_map: np.ndarray, partition: Partition, options: dict = None):
    """
    Computes local credible intervals as decribed in Cai et al.
    """
    # initialize an lci-object and feed it the problem object
    lci_computer = LCIComputer(alpha=alpha, model=model, x_map=x_map, partition=partition, options=options)
    # compute the LCIs
    lci_array = lci_computer.compute()
    return lci_array