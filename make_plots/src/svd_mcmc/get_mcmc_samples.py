
from jax import random
import numpy as np
import ray
from time import time

from uq4pk_src import model_grids,svd_mcmc
from uq4pk_fit.operators import OrnsteinUhlenbeck
from ..mock import load_experiment_data
from .parameters import NUM_SAMPLES, NUM_BURNIN, DATA, LMD_MIN, LMD_MAX, DV, REGPARAM


def get_mcmc_samples(mode: str, sampling: str, q: int = None):
    """
    Creates samples using either SVD-MCMC or HMC, depending on the parameter `sampling`.
    """
    if sampling == "svdmcmc":
        test_burnin = 50
        test_nsample = 100
        base_burnin = 1000
        base_nsample = 2000
    elif sampling == "hmc":
        test_burnin = 50
        test_nsample = 100
        base_burnin = 1000
        base_nsample = 2000
    else:
        raise NotImplementedError("Unknown 'sampling'.")

    if mode == "test":
        burnin_beta_tilde = test_burnin
        nsample_beta_tilde = test_nsample
    elif mode == "base":
        burnin_beta_tilde = base_burnin
        nsample_beta_tilde = base_nsample
    else:
        burnin_beta_tilde = NUM_BURNIN
        nsample_beta_tilde = NUM_SAMPLES

    data = load_experiment_data(DATA)
    theta_true = data.theta_ref
    y_sum = np.sum(data.y)
    y = data.y / y_sum
    sigma_y = data.y_sd / y_sum
    y_bar = data.y_bar / y_sum

    # Create ssps-grid.
    ssps = model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    ssps.logarithmically_resample(dv=DV)

    # Setup regularization term.
    snr = np.linalg.norm(y) / np.linalg.norm(sigma_y)
    regularization_parameter = REGPARAM
    sigma_ou = OrnsteinUhlenbeck(m=12, n=53, h=np.array([4., 2.])).cov
    sigma_beta = sigma_ou / regularization_parameter


    # ------------------------------------------------------- SETUP MCMC SAMPLER

    svd_mcmc_sampler = svd_mcmc.SVD_MCMC(ssps=ssps, theta_v_true=theta_true, y=y, ybar=y_bar, sigma_y=sigma_y, dv=DV)

    # ---------------------------------------------------------- RUN THE SAMPLER

    # Set RNG key for reproducibility
    rng_key = random.PRNGKey(np.random.randint(0, 999999))

    # Sample beta_tilde

    if sampling == "svdmcmc":
        svd_mcmc_sampler.set_q(q)
        beta_tilde_model = svd_mcmc_sampler.get_svd_reduced_model(Sigma_f=sigma_beta)
    elif sampling == "hmc":
        beta_tilde_model = svd_mcmc_sampler.get_full_model(Sigma_f=sigma_beta)
    else:
        raise NotImplementedError("Unknown sampling.")

    beta_tilde_sampler = svd_mcmc_sampler.get_mcmc_sampler(beta_tilde_model, num_warmup=burnin_beta_tilde,
                                                           num_samples=nsample_beta_tilde)
    t0 = time()
    beta_tilde_sampler.run(rng_key)
    t1 = time()
    runtime = t1 - t0

    beta_tilde_sampler.print_summary()
    beta_array = beta_tilde_sampler.get_samples()["beta_tilde"]

    # Rescale.
    beta_array = y_sum * beta_array.reshape(-1, 12, 53)

    result = [beta_array, runtime]
    return result


@ray.remote
def get_mcmc_samples_remote(mode: str, sampling: str, q: int = None):
    """
    Simple wrapper in order to use 'get_mcmc_samples' in parallel with RAy.
    """
    result = get_mcmc_samples(mode, sampling, q)
    return result