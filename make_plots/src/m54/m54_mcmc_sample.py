"""
Create MCMC samples using Prashin's script.
"""

import copy
from jax import random
from pathlib import Path
import numpy as np
from ppxf import ppxf

import uq4pk_src
from uq4pk_fit.special_operators import OrnsteinUhlenbeck
from ..util.geometric_median import geometric_median
from .m54_fit_model import m54_setup_operator
from .parameters import THETA_V, BURNIN_ETA_ALPHA, BURNIN_BETA_TILDE, NSAMPLE_ETA_ALPHA, NSAMPLE_BETA_TILDE, \
    SAMPLE_FILE, REGFACTOR, MEDIAN_FILE, YMED, YSAMPLES


rng_key = random.PRNGKey(32743)


def m54_mcmc_sample(mode: str, out: Path, y: np.ndarray, y_sd: np.ndarray):
    # Define reduced settings for test mode.
    if mode == "test":
        burnin_eta_alpha = 50
        nsample_eta_alpha = 50
        burnin_beta_tilde = 50
        nsample_beta_tilde = 100
    elif mode == "base":
        burnin_eta_alpha = 500
        nsample_eta_alpha = 500
        burnin_beta_tilde = 500
        nsample_beta_tilde = 1000
    else:
        burnin_eta_alpha = BURNIN_ETA_ALPHA
        nsample_eta_alpha = NSAMPLE_ETA_ALPHA
        burnin_beta_tilde = BURNIN_BETA_TILDE
        nsample_beta_tilde = NSAMPLE_BETA_TILDE

    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)

    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
    )
    ssps.resample_spectra(m54_data.lmd)
    # normalise the SSP templates to be light-weighted rather than mass-weighted,
    ssps.Xw /= np.sum(ssps.Xw, 0)
    ssps.dv = m54_data.dv
    ssps.speed_of_light = m54_data.speed_of_light

    npix_buffer_mask = 20
    m54_data.mask[:npix_buffer_mask] = False
    m54_data.mask[-npix_buffer_mask:] = False

    templates = ssps.Xw
    galaxy = m54_data.y
    noise = m54_data.noise_level
    velscale = ssps.dv
    start = [0., 30., 0., 0.]
    bounds = [[-500, 500], [3, 300.], [-0.3, 0.3], [-0.3, 0.3]]
    moments = 4
    mask = m54_data.mask

    # final pixel is NAN, breaks PPXF even though this is masked, so remove it here manually
    templates = templates[:-1, :]
    galaxy = galaxy[:-1]
    noise = noise[:-1]
    ppxf_mask = mask[:-1]

    ppxf_fit = ppxf.ppxf(
        templates,
        galaxy,
        noise,
        velscale,
        start=start,
        degree=-1,
        mdegree=21,
        moments=moments,
        bounds=bounds,
        regul=1e-10,
        mask=ppxf_mask
    )

    continuum_distorition = ppxf_fit.mpoly
    # add an extra element to the end of array to account for one that we chopped off earlier
    continuum_distorition = np.concatenate([continuum_distorition, [continuum_distorition[-1]]])
    ssps_corrected = copy.deepcopy(ssps)
    ssps_corrected.Xw = (ssps_corrected.Xw.T * continuum_distorition).T

    theta_v = THETA_V
    y_loc = 1. * y
    y_loc[-1] = y_loc[-2]
    sigma_y = 1. * y_sd
    sigma_y[-1] = sigma_y[-2]
    # Rescale data.
    y_sum = np.sum(y[mask])
    print(f"y_sum = {y_sum}")
    y_loc = y_loc / y_sum
    sigma_y = sigma_y / y_sum

    svd_mcmc = uq4pk_src.svd_mcmc.SVD_MCMC(
        ssps=ssps_corrected,
        Theta_v_true=theta_v,
        y=y_loc,
        sigma_y=sigma_y,
        dv=m54_data.dv,
        mask=mask,
        do_log_resample=False)

    svd_mcmc.set_q(15)

    P1 = OrnsteinUhlenbeck(m=12, n=53, h=np.array([2., 1.]))
    # Adjust regularization parameter.
    snr = np.linalg.norm(y_loc) / np.linalg.norm(sigma_y)
    beta1 = REGFACTOR * snr
    beta_tilde_prior_cov = P1.cov / beta1
    # Sample eta and alpha
    eta_alpha_model = svd_mcmc.get_consistent_eta_alpha_model(Sigma_beta=beta_tilde_prior_cov)
    eta_alpha_sampler = svd_mcmc.get_mcmc_sampler(eta_alpha_model, num_warmup=burnin_eta_alpha,
                                                  num_samples=nsample_eta_alpha)
    eta_alpha_sampler.run(rng_key)
    eta_alpha_samples = eta_alpha_sampler.get_samples()

    beta_tilde_model = svd_mcmc.get_beta_tilde_model(
        eta_alpha_samples=eta_alpha_samples,
        Sigma_beta_tilde=beta_tilde_prior_cov)
    beta_tilde_sampler = svd_mcmc.get_mcmc_sampler(beta_tilde_model, num_warmup=burnin_beta_tilde,
                                                   num_samples=nsample_beta_tilde)
    beta_tilde_sampler.run(rng_key)
    light_weighed_samples = beta_tilde_sampler.get_samples()["beta_tilde"].reshape(-1, 12, 53)

    # Bring samples to the same scale as the MAP.
    light_weighed_samples = y_sum * light_weighed_samples

    # Compute posterior median.
    posterior_median = geometric_median(X=light_weighed_samples)
    print(f"MCMC scale: {posterior_median.max()}")

    # Compute median prediction.
    observation_operator = m54_setup_operator()
    y_med = observation_operator.fwd_unmasked(posterior_median.flatten(), THETA_V)

    # Compute y-samples.
    y_sample_list = [observation_operator.fwd_unmasked(f.flatten(), THETA_V) for f in light_weighed_samples]
    y_samples = np.array(y_sample_list)

    np.save(str(out / MEDIAN_FILE), posterior_median)
    np.save(str(out / YMED), y_med)
    np.save(str(out / SAMPLE_FILE), light_weighed_samples)
    np.save(str(out / YSAMPLES), y_samples)






