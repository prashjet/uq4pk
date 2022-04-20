import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . observation_operator import ObservationOperator
from . samples import Samples
from . plotting import Plotter

class SVD_MCMC:

    def __init__(self,
                 ssps=None,
                 Theta_v_true=None,
                 df=None,
                 y=None,
                 ybar=None,
                 sigma_y=None,
                 dv=30):
        self.ssps = ssps
        n, p = self.ssps.Xw.shape
        self.n = n
        self.p = p
        self.Theta_v_true = Theta_v_true
        self.df = df
        self.op = ObservationOperator(ssps=ssps, dv=dv)
        self.y = y
        self.ybar = ybar
        self.sigma_y = sigma_y
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
        sum_x_j = np.sum(self.X, 0)
        X_tmp = np.sum(self.y) * self.X/sum_x_j
        self.mu = np.mean(X_tmp, 1)
        self.X_tilde = (X_tmp.T - self.mu).T
        self.D = np.diag(sum_x_j/np.sum(self.y))
        self.Dinv = np.diag(np.diag(self.D)**-1)

    def do_svd(self):
        U, Sig, VT = np.linalg.svd(self.X_tilde)
        self.U = U
        self.Sig = Sig
        self.VT = VT

    def set_q(self, q):
        self.q = q
        self.Z = np.dot(self.U[:,:q], np.diag(self.Sig[:q]))
        self.H = self.VT[:q,:]

    def get_mcmc_sampler(self,
                         model,
                         prngkey=0,
                         num_warmup=500,
                         num_samples=500):
        rng_key = random.PRNGKey(prngkey)
        kernel = NUTS(model)
        mcmc_sampler = MCMC(kernel,
                            num_warmup=num_warmup,
                            num_samples=num_samples)
        return mcmc_sampler

    def get_eta_alpha_model(self,
                            sigma_alpha=0.1,
                            sigma_eta=0.1):
        def eta_alpha_model(sigma_alpha=sigma_alpha,
                            sigma_eta=sigma_eta,
                            y_obs=None):
            alpha = numpyro.sample("alpha", dist.Normal(1, sigma_alpha))
            eta = numpyro.sample("eta",
                                 dist.Normal(0, sigma_eta),
                                 sample_shape=(self.q,))
            ybar = alpha*self.mu + jnp.dot(self.Z, eta)
            nrm = dist.Normal(loc=ybar, scale=self.sigma_y)
            y_obs = numpyro.sample("y_obs", nrm, obs=self.y)
            return y_obs
        return eta_alpha_model

    def get_beta_tilde_model(self,
                             eta_alpha_samples=None,
                             Sigma_beta_tilde=None):
        mu_alpha_y = np.mean(eta_alpha_samples['alpha'])
        sigma_alpha_y = np.std(eta_alpha_samples['alpha'])
        mu_eta_y = np.mean(eta_alpha_samples['eta'], 0)
        sigma_eta_y = np.std(eta_alpha_samples['eta'], 0)
        mu_beta_tilde = np.zeros(self.p)
        def beta_tilde_model():
            # hack for non-negativity
            beta_tilde = numpyro.sample("beta_tilde",
                                        dist.TruncatedNormal(0, 10., low=0),
                                        sample_shape=(self.p,))
            # from the datafit
            nrm = dist.Normal(mu_eta_y, sigma_eta_y)
            numpyro.factor("likelihood",
                           jnp.sum(nrm.log_prob(jnp.dot(self.H, beta_tilde))))
            alpha_nrm = dist.Normal(mu_alpha_y, sigma_alpha_y)
            numpyro.factor("normalisation",
                           alpha_nrm.log_prob(jnp.sum(beta_tilde)))
            # prior
            beta_prior = dist.MultivariateNormal(mu_beta_tilde,
                                                 Sigma_beta_tilde)
            numpyro.factor("regulariser", beta_prior.log_prob(beta_tilde))
            return beta_tilde
        return beta_tilde_model

    def get_beta_from_beta_tilde(self, beta_tilde_samples):
        beta = np.dot(self.Dinv, beta_tilde_samples['beta_tilde'].T).T
        return beta

    def get_pp_from_beta_tilde(self, beta_tilde_samples):
        beta = self.get_beta_from_beta_tilde(beta_tilde_samples)
        y_smp = np.dot(self.X, beta.T).T
        return y_smp

    def get_pp_from_beta_tilde_via_lowdim(self, beta_tilde_samples):
        alpha = np.sum(beta_tilde_samples['beta_tilde'], 1)
        eta = np.dot(self.H, beta_tilde_samples['beta_tilde'].T).T
        y_smp = alpha[:,np.newaxis]*self.mu[np.newaxis,:]
        y_smp += np.dot(self.Z, eta.T).T
        return y_smp

    def get_pp_from_eta_alpha(self, eta_alpha_samples):
        y_smp = self.mu[np.newaxis,:] * eta_alpha_samples['alpha'][:,np.newaxis]
        y_smp += np.dot(self.Z, eta_alpha_samples['eta'].T).T
        return y_smp

    def plot_posterior_predictve(self, y_smp, even_tailed_limit=1):
        lo = even_tailed_limit
        hi = 100 - even_tailed_limit
        lo, med, hi = np.percentile(y_smp, [lo, 50, hi], 0)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ln_lmd = self.ssps.w
        ax[0].plot(ln_lmd, med)
        ax[0].fill_between(ln_lmd, lo, hi, alpha=0.5)
        ax[0].plot(ln_lmd, self.y, '.k', ms=1)
        ax[1].plot(ln_lmd, med-med)
        ax[1].fill_between(ln_lmd, lo-med, hi-med, alpha=0.5)
        ax[1].plot(ln_lmd, self.y-med, '.k', ms=1)
        if self.ybar is not None:
            ax[1].plot(ln_lmd, self.ybar-med, '-k', lw=1)
        ax[1].set_xlabel('ln [$\lambda$/Ang.]')
        fig.tight_layout()
        return fig

    def plot_pixel_wise_posterior_percentiles(self,
                                              beta_tilde_samples,
                                              percentile=50,
                                              clim=(0,10)):
        fig, ax = plt.subplots(1, 1)
        beta = self.get_beta_from_beta_tilde(beta_tilde_samples)
        mysmp = Samples(x=beta)
        plotter = Plotter(ssps=self.ssps, df=self.df,  beta_smp=mysmp)
        plotter.plot_df(view='percentile',
                        clim=clim,
                        percentile=percentile,
                        lognorm=False)
        return fig






    # end
