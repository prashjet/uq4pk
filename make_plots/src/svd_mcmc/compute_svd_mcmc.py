
import numpy as np
from pathlib import Path
from time import time

from uq4pk_fit.inference.fcis_from_samples2d import fcis_from_samples2d
from uq4pk_fit.inference import mean_jaccard_distance
from .get_mcmc_samples import get_mcmc_samples
from .parameters import QLIST, HMCSAMPLES, SVDSAMPLES, TIMES, ERRORS


def compute_svd_mcmc(mode: str, out: Path):
    """
    Performs the varying q test. For different values of q, computes the Kullback-Leibler divergence between the
    SVD-MCMC samples and the samples from full HMC.
    """
    _compute_samples(mode, out)
    _compute_error(mode, out)


def _compute_samples(mode: str, out: Path):
    # Create HMC samples.
    hmc_samples = get_mcmc_samples(mode=mode, sampling="hmc")
    # Store samples.
    np.save(str(out / HMCSAMPLES), hmc_samples)

    # Get list of SVD-MCMC samples.
    svd_sample_list = []
    time_list = []
    if mode == "test" or mode == "base":
        q_list = np.arange(3, 30, 3)
    else:
        q_list = QLIST

    # USE RAY TO SPEED THIS UP!!!
    for q in q_list:
        t0 = time()
        svd_samples = get_mcmc_samples(mode=mode, sampling="svdmcmc", q=q)
        t1 = time()
        t = t1 - t0
        time_list.append(t)
        svd_sample_list.append(svd_samples)
        # Store samples and times (for safety reasons, this happens in the loop).
        svd_sample_array = np.array(svd_sample_list)
        np.save(str(out / SVDSAMPLES), svd_sample_array)
        # Store times.
        np.save(str(out / TIMES), np.array(time_list))


def _compute_error(mode: str, out: Path):
    sigma_list = [np.array([0.5, 1.])]
    # Load samples.
    svd_sample_array = np.load(str(out / SVDSAMPLES))
    hmc_samples = np.load(str(out / HMCSAMPLES)).reshape(-1, 12, 53)
    # Estimate FCIs.
    fci_low_hmc, fci_upp_hmc = fcis_from_samples2d(alpha=0.05, samples=hmc_samples, sigmas=sigma_list)
    fci_hmc = np.column_stack([fci_low_hmc.flatten(), fci_upp_hmc.flatten()])
    # Compute errors.
    error_list = []
    for svd_samples in svd_sample_array:
        svd_samples = svd_samples.reshape(-1, 12, 53)
        fci_low_svd, fci_upp_svd = fcis_from_samples2d(alpha=0.05, samples=svd_samples, sigmas=sigma_list)
        fci_svd = np.column_stack([fci_low_svd.flatten(), fci_upp_svd.flatten()])
        error = mean_jaccard_distance(fci_hmc, fci_svd)
        error_list.append(error)

    # Store KL-divergences.
    np.save(str(out / ERRORS), np.array(error_list))



