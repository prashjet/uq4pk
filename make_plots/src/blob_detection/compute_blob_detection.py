"""
Computes the FCI stack necessary used blob detection.
"""

from jax import random
import numpy as np
import pandas
from pathlib import Path
from time import time

from uq4pk_fit.inference import StatModel
from uq4pk_fit.inference import LightWeightedForwardOperator, fcis_from_samples2d
import uq4pk_src
from ..mock import load_experiment_data
from .parameters import SIGMA_LIST, DATA, LOWER_STACK, UPPER_STACK, DV, LMD_MIN, LMD_MAX, MAP, \
    SPEEDUP_OPTIONS, LOWER_STACK_SPEEDUP, UPPER_STACK_SPEEDUP, TIMES, REGFACTOR
from uq4pk_src import model_grids,svd_mcmc
from uq4pk_fit.special_operators import OrnsteinUhlenbeck
from ..mock import SimulatedExperimentData


def compute_blob_detection(out: Path, mode: str):
    """
    Computes FCI stack both with speedup and without, writes output in "out".
    """
    # First, compute with speedup.
    t0 = time()
    #_compute_fcis(out=out, mode=mode, speed_up=True)
    t1 = time()
    t_speedup = t1 - t0
    print(f"TIME WITH SPEEDUP: {t_speedup} s.")
    # Then, compute exactly.
    _compute_fcis_with_mcmc(out=out, mode=mode)
    t2 = time()
    t_exact = t2 - t1
    print(f"TIME WITHOUT SPEEDUP: {t_exact} s.")
    times = np.array([t_speedup, t_exact]).reshape(1, 2)
    times_frame = pandas.DataFrame(data=times, columns=["MCMC", "optimization"])
    times_frame.to_csv(out / TIMES)


def _compute_fcis_with_mcmc(out: Path, mode: str):
    data = load_experiment_data(DATA)
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    if mode == "test":
        burnin_beta_tilde = 50
        nsample_beta_tilde = 100
    elif mode == "base":
        burnin_beta_tilde = 500
        nsample_beta_tilde = 1000
    else:
        burnin_beta_tilde = 10000
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
    sigma_ou = OrnsteinUhlenbeck(m=12, n=53, h=np.array([4., 2.])).cov
    sigma_beta = sigma_ou / regularization_parameter


    # ------------------------------------------------------- SETUP MCMC SAMPLER

    svd_mcmc_sampler = svd_mcmc.SVD_MCMC(ssps=ssps, Theta_v_true=theta_true, y=y, ybar=y_bar, sigma_y=sigma_y, dv=DV)
    # Choose degrees of freedom for reduced problem.
    svd_mcmc_sampler.set_q(15)

    # ---------------------------------------------------------- RUN THE SAMPLER

    # Set RNG key for reproducibility
    rng_key = random.PRNGKey(32743)

    # Sample beta_tilde
    beta_tilde_model = svd_mcmc_sampler.get_beta_tilde_dr_single_model(Sigma_beta_tilde=sigma_beta)
    beta_tilde_sampler = svd_mcmc_sampler.get_mcmc_sampler(beta_tilde_model, num_warmup=burnin_beta_tilde,
                                                           num_samples=nsample_beta_tilde)
    beta_tilde_sampler.run(rng_key)
    beta_tilde_sampler.print_summary()
    beta_array = beta_tilde_sampler.get_samples()["beta_tilde"]

    # Reshape array into image format.
    beta_array = y_sum * beta_array.reshape(-1, 12, 53)

    alpha = 0.05
    lower_stack, upper_stack = fcis_from_samples2d(alpha=alpha, samples=beta_array, sigmas=SIGMA_LIST)
    np.save(file=str(out / LOWER_STACK), arr=lower_stack)
    np.save(file=str(out / UPPER_STACK), arr=upper_stack)
