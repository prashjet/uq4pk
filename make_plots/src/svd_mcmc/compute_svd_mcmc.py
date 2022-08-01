
import numpy as np
from pathlib import Path
import ray

from uq4pk_fit.inference.fcis_from_samples2d import fcis_from_samples2d
from uq4pk_fit.inference import mean_jaccard_distance
from .get_mcmc_samples import get_mcmc_samples, get_mcmc_samples_remote
from .parameters import QLIST, HMCSAMPLES, HMCCONTROL, SVDSAMPLES, TIMES, ERRORS


PARALLEL = True     # Toggles parallelization.
NUM_CPUS = 7        # Number of CPUs used for parallel computation.


def compute_svd_mcmc(mode: str, out: Path):
    """
    Performs the varying q test. For different values of q, computes the Kullback-Leibler divergence between the
    SVD-MCMC samples and the samples from full HMC.
    """
    #_compute_samples(mode, out)
    _compute_error(mode, out)


def _compute_samples(mode: str, out: Path):
    """
    Computes samples for q in QLIST. The computations can be parallelized using Ray.
    """
    if mode == "test":
        q_list = QLIST
    elif mode == "base":
        q_list = QLIST
    else:
        q_list = QLIST
    if PARALLEL:
        _compute_samples_parallel(mode, out, q_list)
    else:
        _compute_samples_sequential(mode, out, q_list)


def _compute_samples_parallel(mode: str, out: Path, q_list):
    # Initialize Ray
    ray.shutdown()
    ray.init(num_cpus=7)
    # Perform computations in parallel.
    print("Collecting IDs...")
    ray_ids = [get_mcmc_samples_remote.remote(mode=mode, sampling="svdmcmc", q=q) for q in q_list]
    # Also add IDs for HMC samples.
    ray_ids.append(get_mcmc_samples_remote.remote(mode=mode, sampling="hmc"))
    ray_ids.append(get_mcmc_samples_remote.remote(mode=mode, sampling="hmc"))
    print(f"Number of jobs: {len(ray_ids)}.")
    print("Starting computations...")
    result_list = ray.get(ray_ids)
    print("Extracting results...")
    # Extract results.
    svd_sample_list = [result[0] for result in result_list[:-2]]
    time_list = [result[1] for result in result_list[:-2]]
    # Store results
    hmc_samples = result_list[-2][0]
    hmc_control = result_list[-1][0]
    hmc_time = result_list[-1][1]
    time_list.append(hmc_time)
    svd_sample_array = np.array(svd_sample_list)
    np.save(str(out / HMCSAMPLES), hmc_samples)
    np.save(str(out / HMCCONTROL), hmc_control)
    np.save(str(out / SVDSAMPLES), svd_sample_array)
    np.save(str(out / TIMES), np.array(time_list))


def _compute_samples_sequential(mode: str, out: Path, q_list):
    # Get list of SVD-MCMC samples.
    svd_sample_list = []
    time_list = []

    for q in q_list:
        svd_samples, t = get_mcmc_samples(mode=mode, sampling="svdmcmc", q=q)
        time_list.append(t)
        svd_sample_list.append(svd_samples)
        # Store samples and times (for safety reasons, this happens in the loop).
        svd_sample_array = np.array(svd_sample_list)
        np.save(str(out / SVDSAMPLES), svd_sample_array)
        # Store times.
        np.save(str(out / TIMES), np.array(time_list))
    # Also create HMC samples.
    hmc_samples, t_hmc = get_mcmc_samples(mode=mode, sampling="hmc")
    # Store samples.
    np.save(str(out / HMCSAMPLES), hmc_samples)
    # Repeat for control.
    hmc_control, t_control = get_mcmc_samples(mode=mode, sampling="hmc")
    time_list.append(t_control)
    np.save(str(out / TIMES), np.array(time_list))
    np.save(str(out / HMCCONTROL), hmc_control)


def _compute_error(mode: str, out: Path):
    sigma_list = [5 * np.array([0.5, 1.])]
    # Load samples.
    svd_sample_array = np.load(str(out / SVDSAMPLES))
    hmc_samples = np.load(str(out / HMCSAMPLES)).reshape(-1, 12, 53)
    hmc_control = np.load(str(out / HMCCONTROL)).reshape(-1, 12, 53)
    # Estimate FCIs.
    fci_low_hmc, fci_upp_hmc = fcis_from_samples2d(alpha=0.05, samples=hmc_samples, sigmas=sigma_list)
    fci_hmc = np.column_stack([fci_low_hmc.flatten(), fci_upp_hmc.flatten()])
    # Compute errors.
    error_list = []
    for svd_samples in svd_sample_array:
        svd_samples = svd_samples.reshape(-1, 12, 53)
        fci_low_svd, fci_upp_svd = fcis_from_samples2d(alpha=0.05, samples=svd_samples, sigmas=sigma_list)
        fci_svd = np.column_stack([fci_low_svd.flatten(), fci_upp_svd.flatten()])
        error = mean_jaccard_distance(fci_svd, fci_hmc)
        error_list.append(error)
    # Last entry is for control.
    fci_low_control, fci_upp_control = fcis_from_samples2d(alpha=0.05, samples=hmc_control, sigmas=sigma_list)
    fci_control = np.column_stack([fci_low_control.flatten(), fci_upp_control.flatten()])
    mc_error = mean_jaccard_distance(fci_control, fci_hmc)
    error_list.append(mc_error)
    # Store errors.
    np.save(str(out / ERRORS), np.array(error_list))


def jaccard_error(fci1, fci2):
    return mean_jaccard_distance(fci1, fci2)


def relative_l2_error(fci1, fci2):
    fci1_low = fci1[:, 0]
    fci1_upp = fci1[:, 1]
    fci2_low = fci2[:, 0]
    fci2_upp = fci2[:, 1]
    error_low = np.linalg.norm(fci1_low - fci2_low) / np.linalg.norm(fci2_low)
    error_upp = np.linalg.norm(fci1_upp - fci2_upp) / np.linalg.norm(fci2_upp)
    return error_low + error_upp


