"""
Contains the class "ExperimentData" and the functions "simulate_data" and "forward".
"""
import numpy as np

from uq4pk_fit.inference import MassWeightedForwardOperator, LightWeightedForwardOperator

from .simulated_experiment_data import SimulatedExperimentData


HERMITE_ORDER = 4


def simulate(name: str, snr: float, ssps, f_im: np.array, theta_v: np.array, light_weighted: bool, dv=10,
             do_log_resample=True, mask=None) -> SimulatedExperimentData:
    """
    Simulates a dataset. Generates a measurement from the provided distribution function, while
    theta_v is fixed to [30, 100, 1, 0, 0, -0.05, 0.1]. Then adds artificial noise to generate the simulated
    noisy measurement.

    :return: ExperimentData
        All the relevant parameters combined in an object of type "ExperimentData".
    """
    hermite_order = theta_v.size - 3
    if light_weighted:
        op = LightWeightedForwardOperator(theta=theta_v, hermite_order=hermite_order, ssps=ssps, dv=dv,
                                          do_log_resample=do_log_resample, mask=mask)
    else:
        op = MassWeightedForwardOperator(hermite_order=hermite_order, ssps=ssps, dv=dv, do_log_resample=do_log_resample, mask=mask)
    # NORMALIZE
    f_im = f_im / np.sum(f_im)
    f_true = f_im.flatten()
    y_bar = op.fwd(f_true, theta_v)
    # the signal-to-noise ratio is defined as the ratio of np.mean(y) / np.mean(abs(xi))
    # next, perturb the measurement by Gaussian noise
    m = y_bar.size
    # noise is scaled to achieve exactly the given signal-to-noise ratio
    sigma = np.linalg.norm(y_bar) / (snr * np.sqrt(m))
    y_sd = sigma * np.ones(m)
    noi = y_sd * np.random.randn(y_bar.size)
    y = y_bar + noi
    y_sd = y_sd
    y = y
    print(f"Data scale: {np.linalg.norm(y)}")
    print(f"Actual snr = {np.linalg.norm(y_bar) / np.linalg.norm(noi)}")
    print(f"||y_exact|| / ||y_sd|| = {np.linalg.norm(y_bar) / np.linalg.norm(y_sd)}")
    experiment_data = SimulatedExperimentData(name=name, snr=snr, y=y, y_sd=y_sd, y_bar=y_bar, f_true=f_true,
                                              f_ref=f_true, theta_true=theta_v, hermite_order=HERMITE_ORDER)
    return experiment_data