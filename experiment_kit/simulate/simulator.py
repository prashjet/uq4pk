"""
Contains the class "ExperimentData" and the functions "simulate" and "forward".
"""
import numpy as np

from uq4pk_src.distribution_function import RandomGMM_DistributionFunction
from uq4pk_fit.inference import *

from experiment_kit.experiment.experiment_data import ExperimentData
from .sample_theta import sample_theta


def simulate(name: str, snr, f_im=None, theta_noise=0.05, downsampling: int=1) -> ExperimentData:
    """
    Simulates a dataset. Generates a measurement from the provided distribution function, while
    theta_v is fixed to [30, 100, 1, 0, 0, -0.05, 0.1]. Then adds artificial noise to generate the simulated
    noisy measurement.
    :param snr: float > 0
        The desired signal-to-noise ratio. The signal-to-noise ratio is defined as
        snr = ||y||/||y_noi - y||,
        where y is the noise-free and y_noi is the noisy measurement.
        The noise is scaled in such a way that this signal-to-noise ratio
        holds exactly.
    :param f_im: ndarray, optional
        If provided, the measurements are simulated with this distribution function as ground truth.
        Otherwise, a distribution function is simulated as a random Gaussian mixture.
    :return: ExperimentData
        All the relevant parameters combined in an object of type "ExperimentData".
    """
    if downsampling != 1:
        op = UpsamplingForwardOperator(scale=downsampling)
    else:
        op = ForwardOperator()
    if f_im is None:
        # if no f is provided, we simulate one.
        f_im = RandomGMM_DistributionFunction(modgrid=op.modgrid).F
    # downsample
    f_im = op.downsample(f_im)
    # NORMALIZE
    f_im = f_im / np.sum(f_im)
    f_true = f_im.flatten()
    theta_guess = np.array([30, 100, 1., 0., 0., -0.05, 0.1])
    theta_v, theta_sd = sample_theta(q=theta_noise, theta_v=theta_guess)
    y_exact = op.fwd(f_true, theta_v)
    # the signal-to-noise ratio is defined as the ratio of np.mean(y) / np.mean(abs(xi))
    # next, perturb the measurement by Gaussian noise
    m = y_exact.size
    sigma = np.linalg.norm(y_exact) / (snr * np.sqrt(m))   # noise is scaled to achieve roughly the given signal-to-noise ratio
    y_sd = sigma * np.ones(m)
    noi = y_sd * np.random.randn(y_exact.size)
    y = y_exact + noi
    snr_true = np.linalg.norm(y_exact) / np.linalg.norm(noi)
    print(f"snr={snr_true}")
    print(f"||y||/||y_noi-y|| = {np.linalg.norm(y)/np.linalg.norm(y - y_exact)}")
    print(f"rdm = {np.linalg.norm((y - op.fwd(f_true, theta_v)) / y_sd) / np.sqrt(y.size)}")
    print(f"sre(theta_guess) = {np.linalg.norm((theta_v -theta_guess)/theta_sd) / np.linalg.norm(theta_v / theta_sd)}")
    print(f"delta = {np.linalg.norm(y - y_exact)}")
    print(f"||y_sd|| = {np.linalg.norm(y_sd)}")
    experiment_data = ExperimentData(name=name, y=y, y_sd=y_sd, f_true=f_true, f_ref=f_true, theta_true=theta_v, theta_guess=theta_guess,
                                     theta_sd=theta_sd, forward_operator=op, theta_noise=theta_noise)
    return experiment_data