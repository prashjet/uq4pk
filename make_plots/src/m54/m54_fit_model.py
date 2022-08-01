import copy
from dataclasses import dataclass
import numpy as np
from ppxf import ppxf

from matplotlib import pyplot as plt
import uq4pk_src
from uq4pk_fit.inference import StatModel, FittedModel, LightWeightedForwardOperator
from uq4pk_fit.special_operators import OrnsteinUhlenbeck
from uq4pk_fit.visualization import plot_distribution_function


@dataclass
class M54Model:
    fitted_model: FittedModel
    forward_operator: LightWeightedForwardOperator
    y_sd: np.ndarray
    theta_v: np.ndarray
    mask: np.ndarray


def m54_setup_operator():
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

    # ------------------------------------ FIT NONLINEAR PARTS USING PPXF

    print("Fit using PPXF...")

    mask = m54_data.mask
    npix_buffer_mask = 20
    mask[:npix_buffer_mask] = False
    mask[-npix_buffer_mask:] = False

    templates = ssps.Xw
    galaxy = m54_data.y
    noise = m54_data.noise_level
    velscale = ssps.dv
    start = [0., 30., 0., 0.]
    bounds = [[-500, 500], [3, 300.], [-0.3, 0.3], [-0.3, 0.3]]
    moments = 4

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

    # Correct templates using fitted polynomials.
    continuum_distorition = ppxf_fit.mpoly
    sol = ppxf_fit.sol
    theta_v = np.array([sol[0], sol[1], 1., 0., 0., -sol[2], sol[3]])

    # add an extra element to the end of array to account for one that we chopped off earlier
    continuum_distorition = np.concatenate([continuum_distorition, [continuum_distorition[-1]]])
    ssps_corrected = copy.deepcopy(ssps)
    ssps_corrected.Xw = (ssps_corrected.Xw.T * continuum_distorition).T

    forward_operator = LightWeightedForwardOperator(hermite_order=4, mask=mask, ssps=ssps_corrected,
                                                    dv=ssps_corrected.dv,
                                                    do_log_resample=False, theta=theta_v)
    return forward_operator



def m54_fit_model(y: np.ndarray, y_sd: np.ndarray, regparam: float) -> M54Model:
    """

    :param y: UNMASKED data.
    :param y_sd: UNMASKED data noise levels.
    :param theta_v: Assumed value of theta_v.
    :return:
    """
    # Get forward operator.
    forward_operator = m54_setup_operator()
    # Get mask.
    mask = forward_operator.mask
    y_masked = y[mask]
    y_sd_masked = y_sd[mask]
    snr = np.linalg.norm(y_masked) / np.linalg.norm(y_sd_masked)
    print(f"DATA SCALE = {np.sum(y_masked)}")
    print(f"DATA SNR = {snr}")

    theta_v = forward_operator.theta_v
    # Fit the model
    model = StatModel(y=y_masked, y_sd=y_sd_masked, forward_operator=forward_operator)
    model.P1 = OrnsteinUhlenbeck(m=model.m_f, n=model.n_f, h=np.array([2., 1.]))
    model.fix_theta_v(indices=np.arange(model.dim_theta), values=theta_v)
    model.beta1 = regparam
    fitted_model = model.fit()

    fig, ax = plt.subplots(1, 1)
    plot_distribution_function(ax=ax, image=fitted_model.f_map, ssps=forward_operator.ssps)
    plt.show()

    m54_model = M54Model(fitted_model=fitted_model, forward_operator=forward_operator,
                         y_sd=y_sd, theta_v=theta_v, mask=mask)

    return m54_model