
import numpy as np

from ..ci_computer import compute_credible_intervals
from ..linear_model import LinearModel
from ..partition import Partition
from .partition_to_evaluation_map import partition_to_evaluation_map


def lci(alpha: float,  model: LinearModel, x_map: np.ndarray, partition: Partition, options: dict = None):
    """
    Computes local credible intervals as decribed in Cai et al.

    :param alpha: The credibility parameter. For example, alpha = 0.05 corresponds to 95%-credibility.
    :param model: Defines the (Bayesian) linear statistical model.
    :param x_map: The MAP estimate corresponding to ``model``.
    :param partition: The partition of the parameter space.
    :param options: A dictionary with additional options.
            - "use_ray": If True, then the computation is parallelized with the Ray framework.
            - "num_cpus": Number of CPUs used by Ray.
            - "solver": The optimization solver. "slsqp" for SLSQP, "ecos" for ECOS solver.
    """
    # Generate an affine evaluation map from the partition
    affine_evaluation_map = partition_to_evaluation_map(partition, x_map)
    # Compute the local credible intervals
    lci_arr = compute_credible_intervals(alpha=alpha, model=model, x_map=x_map, aemap=affine_evaluation_map,
                                         options=options)
    # The array is already in x_space and can be returned without further ado.
    return lci_arr