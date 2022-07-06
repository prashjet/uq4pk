
from jax import random
import numpy as np

from uq4pk_src import model_grids,svd_mcmc
from uq4pk_fit.special_operators import OrnsteinUhlenbeck
from ..mock import load_experiment_data
from .parameters import NUM_SAMPLES, NUM_BURNIN, DATA, LMD_MIN, LMD_MAX, DV, REGFACTOR


def get_mcmc_samples(mode: str, sampling: str, q: int = None) -> np.ndarray:
    """
    Creates samples using either SVD-MCMC or HMC.

    :returns: Array of shape (n, d), where n is the number of samples and d is the dimension.
    """
    if sampling == "svdmcmc":
        test_burnin = 500
        test_nsample = 1000
        base_burnin = 1000
        base_nsample = 1000
    elif sampling == "hmc":
        test_burnin = 500
        test_nsample = 1000
        base_burnin = 500
        base_nsample = 500
    else:
        raise NotImplementedError("Unknown 'sampling'.")

    if mode == "test":
        sampling = "svdmcmc"
        q = 15
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
    regularization_parameter = REGFACTOR * snr
    sigma_ou = OrnsteinUhlenbeck(m=12, n=53, h=np.array([4., 2.])).cov
    sigma_beta = sigma_ou / regularization_parameter


    # ------------------------------------------------------- SETUP MCMC SAMPLER

    svd_mcmc_sampler = svd_mcmc.SVD_MCMC(ssps=ssps, Theta_v_true=theta_true, y=y, ybar=y_bar, sigma_y=sigma_y, dv=DV)

    # ---------------------------------------------------------- RUN THE SAMPLER

    # Set RNG key for reproducibility
    rng_key = random.PRNGKey(32743)

    # Sample beta_tilde

    if sampling == "svdmcmc":
        svd_mcmc_sampler.set_q(q)
        beta_tilde_model = svd_mcmc_sampler.get_beta_tilde_dr_single_model(Sigma_beta_tilde=sigma_beta)
    elif sampling == "hmc":
        beta_tilde_model = svd_mcmc_sampler.get_beta_tilde_direct_model(Sigma_beta_tilde=sigma_beta)
    else:
        raise NotImplementedError("Unknown sampling.")

    beta_tilde_sampler = svd_mcmc_sampler.get_mcmc_sampler(beta_tilde_model, num_warmup=burnin_beta_tilde,
                                                           num_samples=nsample_beta_tilde)
    beta_tilde_sampler.run(rng_key)
    beta_tilde_sampler.print_summary()
    beta_array = beta_tilde_sampler.get_samples()["beta_tilde"]

    # Rescale.
    beta_array = y_sum * beta_array

    return beta_array