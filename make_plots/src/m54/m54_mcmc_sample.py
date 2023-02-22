
import copy
from jax import random
from pathlib import Path
import numpy as np
from ppxf import ppxf
from timeit import default_timer as timer

import uq4pk_src
from uq4pk_fit.operators import OrnsteinUhlenbeck
from .m54_fit_model import m54_setup_operator
from .parameters import SVDMCMC_BURNIN, SVDMCMC_NSAMPLES, SAMPLES_SVDMCMC, MEAN_SVDMCMC, YMEAN_SVDMCMC, \
    YSAMPLES_SVDMCMC, HMC_NSAMPLES, HMC_BURNIN, SAMPLES_HMC, YSAMPLES_HMC, YMEAN_HMC, MEAN_HMC, MASK


rng_key = random.PRNGKey(32743)


def m54_mcmc_sample(mode: str, out: Path, y: np.ndarray, y_sd: np.ndarray, sampling: str, regparam: float) -> float:
    """
    Performs the MCMC simulation for the M54 data. The generated samples are stored as .npy-file.
    """
    # Distinguish testing-setups depending on sampler.
    if sampling == "svdmcmc":
        test_burnin = 50
        test_nsamples = 100
        base_burnin = 500
        base_nsamples = 1000
        final_burnin = SVDMCMC_BURNIN
        final_nsamples = SVDMCMC_NSAMPLES
        mean_file = MEAN_SVDMCMC
        ymean_file = YMEAN_SVDMCMC
        sample_file = SAMPLES_SVDMCMC
        ysample_file = YSAMPLES_SVDMCMC
    elif sampling == "hmc":
        test_burnin = 50
        test_nsamples = 100
        base_burnin = 500
        base_nsamples = 1000
        final_burnin = HMC_BURNIN
        final_nsamples = HMC_NSAMPLES
        mean_file = MEAN_HMC
        ymean_file = YMEAN_HMC
        sample_file = SAMPLES_HMC
        ysample_file = YSAMPLES_HMC
    else:
        raise NotImplementedError("Unknown sampler.")

    # Define reduced settings for test mode.
    if mode == "test":
        sampling = "svdmcmc"    # Cannot test full HMC.
        burnin = test_burnin
        nsample = test_nsamples
    elif mode == "base":
        burnin = base_burnin
        nsample = base_nsamples
    else:
        burnin = final_burnin
        nsample = final_nsamples

    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)

    ssps = uq4pk_src.model_grids.MilesSSP(
        miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
        imf_string='Ebi1.30',
        lmd_min=None,
        lmd_max=None,
        age_lim=(0.1, 14)
    )
    ssps.resample_spectra(m54_data.lmd)
    # normalise the SSP templates to be light-weighted rather than mass-weighted,
    ssps.Xw /= np.sum(ssps.Xw, 0)
    ssps.dv = m54_data.dv
    ssps.speed_of_light = m54_data.speed_of_light
    m_f = 12
    n_f = 46

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
        regul=1e-11,
        mask=ppxf_mask
    )

    continuum_distorition = ppxf_fit.mpoly
    # add an extra element to the end of array to account for one that we chopped off earlier
    continuum_distorition = np.concatenate([continuum_distorition, [continuum_distorition[-1]]])
    ssps_corrected = copy.deepcopy(ssps)
    ssps_corrected.Xw = (ssps_corrected.Xw.T * continuum_distorition).T

    sol = ppxf_fit.sol
    theta_v = np.array([sol[0], sol[1], 1., 0., 0., -sol[2], sol[3]])
    y_loc = 1. * y
    y_loc[-1] = y_loc[-2]
    sigma_y = 1. * y_sd
    sigma_y[-1] = sigma_y[-2]
    # Rescale data.
    y_sum = np.sum(y[mask])
    print(f"y_sum = {y_sum}")
    y_loc = y_loc / y_sum
    sigma_y = sigma_y / y_sum

    svd_mcmc = uq4pk_src.svd_mcmc.SVD_MCMC(ssps=ssps_corrected, theta_v_true=theta_v, y=y_loc, sigma_y=sigma_y,
                                           dv=m54_data.dv, mask=mask, do_log_resample=False)

    P = OrnsteinUhlenbeck(m=m_f, n=n_f, h=np.array([2., 1.]))
    # Adjust regularization parameter.
    beta = regparam
    f_prior_cov = P.cov / beta

    # Prepare samples.
    if sampling == "svdmcmc":
        svd_mcmc.set_q(15)
        mcmc_model = svd_mcmc.get_svd_reduced_model(Sigma_f=f_prior_cov)
    elif sampling == "hmc":
        mcmc_model = svd_mcmc.get_full_model(Sigma_f=f_prior_cov)
    else:
        raise NotImplementedError("Unknown sampler.")

    mcmc_sampler = svd_mcmc.get_mcmc_sampler(mcmc_model, num_warmup=burnin,
                                             num_samples=nsample)
    # Run.
    start = timer()
    mcmc_sampler.run(rng_key)
    end = timer()
    time_for_sampling = end - start
    mcmc_sampler.print_summary()
    light_weighed_samples = mcmc_sampler.get_samples()["f_tilde"].reshape(-1, m_f, n_f)

    # Bring samples to the same scale as the MAP.
    light_weighed_samples = y_sum * light_weighed_samples

    # Compute posterior median.
    posterior_mean = np.mean(light_weighed_samples, axis=0)
    print(f"MCMC scale: {posterior_mean.max()}")

    # Compute y-samples.
    observation_operator = m54_setup_operator()
    y_sample_list = [observation_operator.fwd_unmasked(f.flatten()) for f in light_weighed_samples]
    y_samples = np.array(y_sample_list)

    # Compute mean prediction.
    y_mean = np.mean(y_samples, axis=0)

    np.save(str(out / mean_file), posterior_mean)
    np.save(str(out / ymean_file), y_mean)
    np.save(str(out / sample_file), light_weighed_samples)
    np.save(str(out / ysample_file), y_samples)

    # Also store mask
    np.save(str(out / MASK), mask)

    # Return time for sampling.
    return time_for_sampling




