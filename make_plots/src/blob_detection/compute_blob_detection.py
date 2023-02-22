"""
Computes the FCI stack necessary used blob detection.
"""

from jax import random
import numpy as np
from pathlib import Path

from uq4pk_src import model_grids,svd_mcmc
from uq4pk_fit.operators import OrnsteinUhlenbeck
from uq4pk_fit.statistical_modeling import StatModel, LightWeightedForwardOperator
from uq4pk_fit.filtered_credible_intervals import fcis_from_samples2d
from ..mock import load_experiment_data
from .parameters import SIGMA_LIST, DATA, LOWER_STACK, UPPER_STACK, DV, LMD_MIN, LMD_MAX, REGFACTOR, MAP


def compute_blob_detection(out: Path, mode: str):
    """
    Computes FCI stack both with speedup and without, writes output in "out".
    """
    # Then, compute exactly.
    data = load_experiment_data(DATA)
    if mode == "test":
        burnin_beta_tilde = 50
        nsample_beta_tilde = 100
    elif mode == "base":
        burnin_beta_tilde = 500
        nsample_beta_tilde = 1000
    else:
        burnin_beta_tilde = 5000
        nsample_beta_tilde = 10000

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
    regop = OrnsteinUhlenbeck(m=12, n=53, h=np.array([4., 2.]))
    sigma_ou = regop.cov
    sigma_beta = sigma_ou / regularization_parameter

    # Create MAP.
    fwd_op = LightWeightedForwardOperator(theta=theta_true, ssps=ssps, dv=DV, do_log_resample=True)
    stat_model = StatModel(y=data.y, y_sd=data.y_sd, forward_operator=fwd_op)
    stat_model.beta = regularization_parameter
    stat_model.P = regop
    f_map = stat_model.compute_map()

    # ------------------------------------------------------- SETUP MCMC SAMPLER

    svd_mcmc_sampler = svd_mcmc.SVD_MCMC(ssps=ssps, theta_v_true=theta_true, y=y, ybar=y_bar, sigma_y=sigma_y, dv=DV)
    # Choose degrees of freedom for reduced problem.
    svd_mcmc_sampler.set_q(15)

    # ---------------------------------------------------------- RUN THE SAMPLER

    # Set RNG key for reproducibility
    rng_key = random.PRNGKey(32743)

    # Sample beta_tilde
    beta_tilde_model = svd_mcmc_sampler.get_svd_reduced_model(Sigma_f=sigma_beta)
    beta_tilde_sampler = svd_mcmc_sampler.get_mcmc_sampler(beta_tilde_model, num_warmup=burnin_beta_tilde,
                                                           num_samples=nsample_beta_tilde)
    beta_tilde_sampler.run(rng_key)
    beta_tilde_sampler.print_summary()
    beta_array = beta_tilde_sampler.get_samples()["f_tilde"]

    # Reshape array into image format.
    beta_array = y_sum * beta_array.reshape(-1, 12, 53)

    alpha = 0.05
    lower_stack, upper_stack = fcis_from_samples2d(alpha=alpha, samples=beta_array, sigmas=SIGMA_LIST)
    np.save(file=str(out / LOWER_STACK), arr=lower_stack)
    np.save(file=str(out / UPPER_STACK), arr=upper_stack)
    np.save(file=str(out / MAP), arr=f_map)