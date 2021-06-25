"""
Contains the class "ExperimentData" and the functions "simulate" and "forward".
"""
import numpy as np

from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction


class ExperimentData:
    """
    Assembles all relevant parameters for a simulated experiment.
    :attr y: the exact measurement
    :attr y_noi: the noisy measurement
    :attr f: the true distribution function from which y is generated
    :attr theta_v: the true value of theta_v from which y is generated
    :attr sdev: the standard deviation of the noise in y_noi
    :attr grid: the sampling grid for the distribution function (see ObservationOperator.ssps)
    """
    def __init__(self, y, y_noi, f, theta_v, sdev, grid):
        self.y = y
        self.y_noi = y_noi
        self.f = f
        self.theta_v = theta_v
        self.sdev = sdev
        self.grid = grid


def simulate(snr, f=None) -> ExperimentData:
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
    :param f: ndarray, optional
        If provided, the measurements are simulated with this distribution function as ground truth.
        Otherwise, a distribution function is simulated as a random Gaussian mixture.
    :return: ExperimentData
        All the relevant parameters combined in an object of type "ExperimentData".
    """
    op = ObservationOperator(max_order_hermite=4)
    if f is None:
        # if no f is provided, we simulate one.
        f = RandomGMM_DistributionFunction(modgrid=op.ssps).F
    theta_v = np.array([30, 100, 1., 0., 0., -0.05, 0.1])
    y = op.evaluate(f, theta_v)
    # the signal-to-noise ratio is defined as the ratio of np.mean(y) / np.mean(abs(xi))
    # next, perturb the measurement by Gaussian noise
    sigma = np.mean(y) / snr    # noise is scaled to achieve (roughly) the given signal to noise ratio
    noi = sigma * np.random.randn(y.size)
    y_noi = y + noi
    print(f"snr={snr}")
    print(f"||y||/||y_noi-y|| = {np.linalg.norm(y)/np.linalg.norm(y_noi - y)}")
    experiment_data = ExperimentData(y=y, y_noi=y_noi, f=f, theta_v=theta_v, sdev=sigma, grid=op.ssps.w)
    return experiment_data


def forward(f, theta_v):
    """
    Computes the measurement associated to f and theta_v
    :param f: ndarray
        The distribution function
    :param theta_v: ndarray
    :return: ndarray
        The exact light spectrum associated to f and theta_v
    """
    op = ObservationOperator(max_order_hermite=4)
    return op.evaluate(f, theta_v)