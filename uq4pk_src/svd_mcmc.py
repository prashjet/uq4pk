
import numpy as np
import os


NUM_CPU = 1         # Number of CPUs used for computations.
# Enforce parallel usage of CPU.
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NUM_CPU}"

# Numpyro and jax have to be imported AFTER xla flag is set.
import numpyro
import jax

# Get number of available CPUs.
numpyro.set_platform("cpu")
NUM_CHAINS = jax.local_device_count()
print(f"Using {NUM_CHAINS} CPUs for parallel sampling.")
from jax.lib import xla_bridge
print(f"JAX device: {xla_bridge.default_backend()}")


import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
numpyro.set_platform("cpu")
from numpyro.infer import MCMC, NUTS

from . observation_operator import ObservationOperator


class SVD_MCMC:

    def __init__(self,
                 ssps=None,
                 theta_v_true=None,
                 df=None,
                 y=None,
                 ybar=None,
                 sigma_y=None,
                 dv=30,
                 mask=None,
                 do_log_resample=True):
        self.ssps = ssps
        n, p = self.ssps.Xw.shape
        self.n = n
        self.p = p
        self.Theta_v_true = theta_v_true
        self.df = df
        self.op = ObservationOperator(ssps=ssps,
                                      dv=dv,
                                      do_log_resample=do_log_resample)
        self.y = y
        self.ybar = ybar
        self.sigma_y = sigma_y
        self.mask = mask
        self.preprocess_ssp_templates()
        self.whiten_X()
        self.do_svd()

    def preprocess_ssp_templates(self):
        m = self.ssps.F_tilde_s.shape[0]
        FXw = np.reshape(self.ssps.F_tilde_s * self.ssps.delta_zt, (m,-1))
        V, sigma, h, M = self.op.unpack_Theta_v(self.Theta_v_true)
        F_losvd = self.op.losvd.evaluate_fourier_transform(
            self.op.H_coeffs,
            V,
            sigma,
            h,
            M,
            self.op.omega)
        FXw_conv = (FXw.T * F_losvd).T
        self.X = np.fft.irfft(FXw_conv, self.op.ssps.n_fft, axis=0)

    def whiten_X(self):
        if self.mask is None:
            sum_x_j = np.sum(self.X, 0)
            sum_y = np.sum(self.y)
        else:
            sum_x_j = np.sum(self.X[self.mask, :], 0)
            sum_y = np.sum(self.y[self.mask])
        X_tmp = sum_y * self.X/sum_x_j
        self.mu = np.mean(X_tmp, 1)
        self.X_tilde = (X_tmp.T - self.mu).T
        self.D = np.diag(sum_x_j/sum_y)
        self.Dinv = np.diag(sum_y/sum_x_j)
        self.X_lw = np.dot(self.X, self.Dinv)

    def do_svd(self):
        U, Sig, VT = np.linalg.svd(self.X_tilde)
        self.U = U
        self.Sig = Sig
        self.VT = VT

    def set_q(self, q):
        self.q = q
        self.Z = np.dot(self.U[:, :q], np.diag(self.Sig[:q]))
        self.H = self.VT[:q, :]

    def get_mcmc_sampler(self,
                         model,
                         prngkey=0,
                         num_warmup=500,
                         num_samples=500):
        rng_key = random.PRNGKey(prngkey)
        kernel = NUTS(model)
        samples_per_chain = np.ceil(num_samples / NUM_CHAINS).astype(int)
        print(f"Initializing {NUM_CHAINS} parallel chains with {num_warmup} warmup samples and "
              f"{samples_per_chain} samples per chain.")
        mcmc_sampler = MCMC(kernel,
                            num_warmup=num_warmup,
                            num_samples=samples_per_chain,
                            num_chains=NUM_CHAINS)
        return mcmc_sampler

    def get_full_model(self, Sigma_f=None):
        mu_f_tilde = np.zeros(self.p)
        if self.mask is None:
            def full_model(Sigma_f=Sigma_f,
                           y_obs=None):
                # hack for non-negativity
                f_tilde = numpyro.sample("f_tilde", dist.TruncatedNormal(0, 10., low=0), sample_shape=(self.p,))
                # prior
                f_prior = dist.MultivariateNormal(mu_f_tilde, Sigma_f)
                numpyro.factor("regulariser", f_prior.log_prob(f_tilde))
                # likelihood
                ybar = jnp.dot(self.X_lw, f_tilde)
                nrm = dist.Normal(loc=ybar, scale=self.sigma_y)
                y_obs = numpyro.sample("y_obs", nrm, obs=self.y)
                return y_obs
        else:
            def full_model(Sigma_f=Sigma_f):
                # hack for non-negativity
                f_tilde = numpyro.sample("f_tilde", dist.TruncatedNormal(0, 10., low=0), sample_shape=(self.p,))
                # prior
                f_prior = dist.MultivariateNormal(mu_f_tilde, Sigma_f)
                numpyro.factor("regulariser", f_prior.log_prob(f_tilde))
                # likelihood
                ybar = jnp.dot(self.X_lw, f_tilde)
                nrm = dist.Normal(loc=ybar, scale=self.sigma_y)
                masked_nrm = nrm.mask(self.mask)
                y_obs = numpyro.sample("y_obs", masked_nrm, obs=self.y)
                return y_obs
        return full_model

    def get_svd_reduced_model(self, Sigma_f=None):
        mu_f_tilde = np.zeros(self.p)
        if self.mask is None:
            def svd_reduced_model(Sigma_f=Sigma_f,
                                  y_obs=None):
                # hack for non-negativity
                f_tilde = numpyro.sample("f_tilde",
                                         dist.TruncatedNormal(0, 10., low=0),
                                         sample_shape=(self.p,))
                # prior
                f_prior = dist.MultivariateNormal(mu_f_tilde, Sigma_f)
                numpyro.factor("regulariser", f_prior.log_prob(f_tilde))
                # likelihood in latent space
                eta = jnp.dot(self.H, f_tilde)
                alpha = jnp.sum(f_tilde)
                # likelihood
                ybar = alpha*self.mu + jnp.dot(self.Z, eta)
                nrm = dist.Normal(loc=ybar, scale=self.sigma_y)
                y_obs = numpyro.sample("y_obs", nrm, obs=self.y)
                return y_obs
        else:
            def svd_reduced_model(Sigma_f=Sigma_f):
                # hack for non-negativity
                f_tilde = numpyro.sample("f_tilde",
                                         dist.TruncatedNormal(0, 10., low=0),
                                         sample_shape=(self.p,))
                # prior
                f_prior = dist.MultivariateNormal(mu_f_tilde, Sigma_f)
                numpyro.factor("regulariser", f_prior.log_prob(f_tilde))
                # likelihood in latent space
                eta = jnp.dot(self.H, f_tilde)
                alpha = jnp.sum(f_tilde)
                # likelihood
                ybar = alpha*self.mu + jnp.dot(self.Z, eta)
                nrm = dist.Normal(loc=ybar, scale=self.sigma_y)
                masked_nrm = nrm.mask(self.mask)
                y_obs = numpyro.sample("y_obs", masked_nrm, obs=self.y)
                return y_obs
        return svd_reduced_model
