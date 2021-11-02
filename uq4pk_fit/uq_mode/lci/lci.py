"""
Contains the function "lci".
"""

import numpy as np

from .lci_computer import LCIComputer
from ..partition import Partition


def lci(alpha: float, cost: callable, costgrad: callable, x_map: np.ndarray, partition: Partition,
        lb: np.ndarray = None):
    """
    Computes local credible intervals as decribed in Cai et al.
    """
    # initialize an lci-object and feed it the problem object
    lci_computer = LCIComputer(alpha=alpha, x_map=x_map, cost=cost, costgrad=costgrad, partition=partition, lb=lb)
    # compute the LCIs
    lci_array = lci_computer.compute()
    return lci_array