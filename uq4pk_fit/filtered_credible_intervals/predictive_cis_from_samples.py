
import numpy as np

from .fci import fci_sampling
from .filter import IdentityFilterFunction


def predictive_cis_from_samples(alpha: float, samples: np.ndarray):
    """
    Given one-dimensional spectra, computes simultaneous credible intervals. This function is only used for the
    M54 posterior predictive plot.
    """
    assert samples.ndim == 2
    k, n = samples.shape
    filter = IdentityFilterFunction(dim=n)
    fci_obj = fci_sampling(alpha=alpha, samples=samples, ffunction=filter)

    return fci_obj.lower, fci_obj.upper