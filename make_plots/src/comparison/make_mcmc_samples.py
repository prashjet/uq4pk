"""
file: comparison/make_mcmc_samples.py
"""


import numpy as np
from pathlib import Path

import uq4pk_src.model_grids
from uq4pk_fit.special_operators import OrnsteinUhlenbeck
from simulate_data import SimulatedExperimentData
from uq4pk_src import svd_mcmc
from jax import random
from src.comparison.parameters import LMD_MIN, LMD_MAX, DV, NSAMPLE_ETA_ALPHA, NSAMPLE_BETA_TILDE, BURNIN_ETA_ALPHA, \
    BURNIN_BETA_TILDE, Q, SIGMA_ALPHA, SIGMA_ETA, SAMPLEFILE, REGFACTOR


def make_mcmc_samples(mode: str, data: SimulatedExperimentData, out: Path):
    if mode == "test":
        burnin_eta_alpha = 500
        nsample_eta_alpha = 500
        burnin_beta_tilde = 50
        nsample_beta_tilde = 100
    elif mode == "base":
        burnin_eta_alpha = 2500
        nsample_eta_alpha = 2500
        burnin_beta_tilde = 500
        nsample_beta_tilde = 1000
    else:
        burnin_eta_alpha = BURNIN_ETA_ALPHA
        nsample_eta_alpha = NSAMPLE_ETA_ALPHA
        burnin_beta_tilde = BURNIN_BETA_TILDE
        nsample_beta_tilde = NSAMPLE_BETA_TILDE

    theta_true = data.theta_ref
    y_sum = np.sum(data.y)
    y = data.y / y_sum
    sigma_y = data.y_sd / y_sum
    y_bar = data.y_bar / y_sum

    # Create ssps-grid.
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    ssps.logarithmically_resample(dv=DV)

    # Setup regularization term.
    snr = np.linalg.norm(y) / np.linalg.norm(sigma_y)
    regularization_parameter = REGFACTOR * snr
    sigma_ou = OrnsteinUhlenbeck(m=12, n=53, h=np.array([4., 2.])).cov
    sigma_beta = sigma_ou / regularization_parameter


    # ------------------------------------------------------- SETUP MCMC SAMPLER

    svd_mcmc_sampler = svd_mcmc.SVD_MCMC(ssps=ssps, Theta_v_true=theta_true, y=y, ybar=y_bar, sigma_y=sigma_y, dv=DV)
    # Choose degrees of freedom for reduced problem.
    svd_mcmc_sampler.set_q(Q)

    # Get alpha sampler.
    eta_alpha_model = svd_mcmc_sampler.get_consistent_eta_alpha_model(Sigma_beta=sigma_beta)
    #eta_alpha_model = svd_mcmc_sampler.get_eta_alpha_model(sigma_alpha=SIGMA_ALPHA, sigma_eta=SIGMA_ETA)
    eta_alpha_sampler = svd_mcmc_sampler.get_mcmc_sampler(eta_alpha_model, num_warmup=burnin_eta_alpha,
                                                          num_samples=nsample_eta_alpha)

    # ---------------------------------------------------------- RUN THE SAMPLER

    # Set RNG key for reproducibility
    rng_key = random.PRNGKey(32743)

    # Sample eta_alpha
    eta_alpha_sampler.run(rng_key)
    eta_alpha_sampler.print_summary()
    eta_alpha_samples = eta_alpha_sampler.get_samples()

    # Sample beta_tilde

    beta_tilde_model = svd_mcmc_sampler.get_beta_tilde_model(eta_alpha_samples=eta_alpha_samples,
                                                             Sigma_beta_tilde=sigma_beta)
    beta_tilde_sampler = svd_mcmc_sampler.get_mcmc_sampler(beta_tilde_model, num_warmup=burnin_beta_tilde,
                                                           num_samples=nsample_beta_tilde)
    beta_tilde_sampler.run(rng_key)
    beta_tilde_sampler.print_summary()
    beta_array = beta_tilde_sampler.get_samples()["beta_tilde"]

    # Reshape array into image format.
    beta_array = y_sum * beta_array.reshape(-1, 12, 53)
    np.save(str(out / SAMPLEFILE), beta_array)






