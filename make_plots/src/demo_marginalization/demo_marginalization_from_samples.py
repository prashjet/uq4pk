import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from uq4pk_fit.inference import LightWeightedForwardOperator, marginal_ci_from_samples
from simulate_data import load_experiment_data
from src.experiment_data.experiment_parameters import LMD_MIN, LMD_MAX, DV
import uq4pk_src.model_grids
from uq4pk_fit.special_operators import OrnsteinUhlenbeck
from uq4pk_src import svd_mcmc
from jax import random


# Make samples.
DATA = Path("../experiment_data/snr1000")
REGPARAM = 1000 * 1e3
Q = 15
SIGMA_ALPHA = 0.1
SIGMA_ETA = 0.1
burnin_eta_alpha = 500
nsample_eta_alpha = 500
burnin_beta_tilde = 50
nsample_beta_tilde = 100


data = load_experiment_data(str(DATA))
y = data.y
y_sd = data.y_sd
theta_true = data.theta_ref
y_bar = data.y_bar

# Create ssps-grid.
ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
ssps.logarithmically_resample(dv=DV)

# Setup regularization term.
regularization_parameter = REGPARAM
sigma_ou = OrnsteinUhlenbeck(m=12, n=53, h=np.array([4., 2.])).cov
sigma_beta = sigma_ou / regularization_parameter


# ------------------------------------------------------- SETUP MCMC SAMPLER

svd_mcmc_sampler = svd_mcmc.SVD_MCMC(ssps=ssps,
                                             Theta_v_true=theta_true,
                                             y=y,
                                             ybar=y_bar,
                                             sigma_y=y_sd)
# Choose degrees of freedom for reduced problem.
svd_mcmc_sampler.set_q(Q)

# Get alpha sampler.
eta_alpha_model = svd_mcmc_sampler.get_eta_alpha_model(sigma_alpha=SIGMA_ALPHA, sigma_eta=SIGMA_ETA)
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
beta_tilde_samples = beta_tilde_sampler.get_samples()

# Store the samples FOR BETA not beta_tilde.
beta_array = svd_mcmc_sampler.get_beta_from_beta_tilde(beta_tilde_samples)

# Reshape array into image format.
mass_samples = beta_array.reshape(-1, 12, 53)

# Rescale samples to light weights.
forward_operator = LightWeightedForwardOperator(ssps=ssps, dv=DV, theta=data.theta_ref)
weights = forward_operator.weights.reshape(12, 53)
samples = mass_samples * weights[np.newaxis, :, :]

# Compute marginalized CIs.
lower, upper = marginal_ci_from_samples(alpha=0.05, axis=0, samples=samples)
# Also sum the mean.
mean = np.mean(samples, axis=0)
marginal_mean = np.sum(mean, axis=0)
# Visualize.
x_span = np.arange(lower.size)
plt.plot(x_span, lower, label="lower")
plt.plot(x_span, upper, label="upper")
plt.plot(x_span, marginal_mean, label="mean")
plt.legend()
plt.show()