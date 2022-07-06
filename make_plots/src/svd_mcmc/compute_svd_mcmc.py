
import numpy as np
from pathlib import Path
from time import time

from uq4pk_src.two_sample_kl_divergence import KLdivergence
from .get_mcmc_samples import get_mcmc_samples
from .parameters import QLIST, HMCSAMPLES, SVDSAMPLES, TIMES, DIVERGENCES


def compute_svd_mcmc(mode: str, out: Path):
    """
    Performs the varying q test. For different values of q, computes the Kullback-Leibler divergence between the
    SVD-MCMC samples and the samples from full HMC.
    """
    _compute_samples(mode, out)
    _compute_divergences(mode, out)


def _compute_samples(mode: str, out: Path):
    # Create HMC samples.
    hmc_samples = get_mcmc_samples(mode=mode, sampling="hmc")
    # Store samples.
    np.save(str(out / HMCSAMPLES), hmc_samples)

    # Get list of SVD-MCMC samples.
    svd_sample_list = []
    time_list = []
    if mode == "test" or mode == "base":
        q_list = np.arange(5, 30, 5)
    else:
        q_list = QLIST
    for q in q_list:
        t0 = time()
        svd_samples = get_mcmc_samples(mode=mode, sampling="svdmcmc", q=q)
        t1 = time()
        t = t1 - t0
        time_list.append(t)
        svd_sample_list.append(svd_samples)

    # Store samples.
    svd_sample_array = np.array(svd_sample_list)
    np.save(str(out / SVDSAMPLES), svd_sample_array)

    # Store times.
    np.save(str(out / TIMES), np.array(time_list))


def _compute_divergences(mode: str, out: Path):
    # Load samples.
    svd_sample_array = np.load(str(out / SVDSAMPLES))
    hmc_samples = np.load(str(out / HMCSAMPLES))
    # Compute KL-divergences.
    divergence_list = []
    for svd_samples in svd_sample_array:
        divergence = KLdivergence(hmc_samples, svd_samples)
        divergence_list.append(divergence)

    # Store KL-divergences.
    np.save(str(out / DIVERGENCES), np.array(divergence_list))



