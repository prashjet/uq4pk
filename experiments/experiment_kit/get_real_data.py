
import numpy as np
from ppxf import ppxf

import uq4pk_src

from uq4pk_fit.inference import *
from .experiment_data import ExperimentData

THETA_NOISE = 0.05
THETA_V = np.array([145, 35, 1., 0., 0., 0.028, -0.23])
THETA_SCALE = np.array([145, 35, 1., 0.01, 0.01, 0.028, 0.23])


def get_real_data(simulate=True, normalize=True, snr=None) -> ExperimentData:
    """
    Makes experiment object from M54 data.
    :param simulate: bool
        If True, does not return real data, but data simulated from ground truth
    :return: ExperimentData
    """
    # get M54 data
    y, y_sd, fwdop, theta_guess, theta_sd, f_gt, mask, ssps = _extract_data()
    f_ppxf = _do_ppxf(y, y_sd, mask, ssps)
    if normalize:
        # normalize data
        y, y_sd, f_gt, f_ppxf = _normalize_data(y=y[mask], y_sd=y_sd[mask], f_gt=f_gt, f_ppxf=f_ppxf, fwdop=fwdop,
                                                theta_guess=theta_guess)
    else:
        y, y_sd = y[mask], y_sd[mask]
    if simulate:
        theta_sim = theta_guess + theta_sd * np.random.randn(theta_guess.size)
        y, y_sd = _simulate(f_gt, theta_sim, y_sd, fwdop, snr)
    else:
        # We don't know true theta
        theta_sim = theta_guess
    # make ExperimentData object and return it
    experiment_data = ExperimentData(y=y, y_sd=y_sd, f_true=f_gt, f_ref=f_ppxf, theta_true=theta_sim,
                                     theta_guess=theta_guess, theta_sd=theta_sd, forward_operator=fwdop, theta_noise=THETA_NOISE)
    return experiment_data


def _extract_data():
    """
    Extracts data from the m54-dataset.
    :returns y: (m,) array_like
        The real measurement data (masked).
    :returns y_sd: (m,) array_like
        The corresponding vector of standard deviations.
    :returns fwdop: ForwardOperator
        The masked forward operator.
    :returns theta_guess: (7,) array
        A good initial guess for the corresponding LOSVD-parameters.
    :returns theta_sd: (7,) array
        The associated prior standard deviations.
    :returns f_gt: (n,) array
        The ground truth as a flattened vector, normalized such that
    :returns f_ppxf: (n,) array
        The MAP estimate from the PPFX software.
    """
    # prepare data
    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)
    # read data and noise level
    y = m54_data.y
    y_sd = m54_data.noise_level
    # get mask
    mask = m54_data.mask
    # remove jumps at start and end
    npix_buffer_mask = 20
    mask[:npix_buffer_mask] = False
    mask[-npix_buffer_mask:] = False
    # get initial guess for theta_v
    theta_guess = THETA_V
    # get standard deviations
    theta_sd = THETA_NOISE * THETA_SCALE
    # get "ground truth"
    f_gt = np.flipud(m54_data.ground_truth.T).flatten()
    # theta_v is just the default guess
    # set up the SSPS grid
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
    # construct the forward operator
    fwdop = ForwardOperator(ssps=ssps, dv=ssps.dv, do_log_resample=False, mask=mask)
    return y, y_sd, fwdop, theta_guess, theta_sd, f_gt, mask, ssps


def _do_ppxf(y, y_sd, mask, ssps):
    templates = ssps.Xw
    velscale = ssps.dv
    start = [0., 30., 0., 0.]
    bounds = [[-500, 500], [3, 300.], [-0.3, 0.3], [-0.3, 0.3]]
    moments = 4  # 6

    templates = templates[:-1, :]
    truncated_mask = mask[:-1]
    galaxy = y[:-1]
    noise = y_sd[:-1]

    ppxf_fit = ppxf.ppxf(
        templates,
        galaxy,
        noise,
        velscale,
        start=start,
        degree=8,
        moments=moments,
        bounds=bounds,
        regul=0,
        mask=truncated_mask
    )
    f = ppxf_fit.weights
    # bring ppxf-estimate in right format
    f_im = np.reshape(f, ssps.par_dims)
    f_im = np.flipud(f_im)
    # have to flip for some reason
    f = f_im.flatten()
    return f


def _simulate(f, theta, y_sd, fwdop, snr):
    # if snr is not None, y_sd gets rescaled to achieve given snr.
    y = fwdop.fwd(f, theta)
    std_noise = np.random.randn(y.size)
    if snr is None:
        y_noisy = y + y_sd * std_noise
    else:
        # if snr is not None, y_sd gets rescaled to achieve given snr.
        y_sd = y_sd * np.linalg.norm(y) / (snr * np.linalg.norm(y_sd))
        y_noisy = y + y_sd * std_noise
    # check
    simulated_snr = np.linalg.norm(y_noisy) / np.linalg.norm(y_sd * std_noise)
    print(f"Simulated SNR = {simulated_snr}")
    return y_noisy, y_sd


def _normalize_data(y, y_sd, f_gt, f_ppxf, theta_guess, fwdop):
    """
    Rescale data so that it fits to a age-metallicity distribution that sums to 1.
    :return:
    """
    # scale f_ppxf to 1
    f_ppxf_norm = f_ppxf / np.sum(f_ppxf)
    # rescale f_gt accordingly
    f_gt_norm = f_gt / np.sum(f_gt)
    # obtain a reference measurement
    y_ref = fwdop.fwd(f_ppxf_norm, theta_guess)
    # compute a scaling factor such that ||y_ref|| = ||y||
    s = np.linalg.norm(y_ref) / np.linalg.norm(y)
    # rescale data and noise
    y_norm = y * s
    y_sd_norm = y_sd * s
    assert np.isclose(np.linalg.norm(y_norm), np.linalg.norm(y_ref))
    # return normalized data
    return y_norm, y_sd_norm, f_gt_norm, f_ppxf_norm




