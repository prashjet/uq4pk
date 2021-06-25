"""
Contains class BasicProblem.
"""

import numpy as np

from uq4pk_src.observation_operator import ObservationOperator
from uq4pk_src.distribution_function import RandomGMM_DistributionFunction

class BasicProblem:
    """
    Sets up the basic, untransformed model.
    """
    def __init__(self, hermite_order=4):
        """
        :param hermite_order: order of the Gauss-Hermite expansion used in the LOSVD model.
        """
        self.op = ObservationOperator(max_order_hermite=hermite_order)

        # find out the dimensions
        ftest = RandomGMM_DistributionFunction(modgrid=self.op.ssps).F
        self.n_f_1, self.n_f_2 = ftest.shape
        self.n_f = ftest.size
        self.n_theta_v = 2 + (hermite_order+1)

        # set the hidden prior for theta_v
        V_bar = np.array([40])
        sd_V = np.array([10])
        sigma_bar = np.array([100.])
        sd_sigma = np.array([10.])
        h_bar = np.zeros(hermite_order+1)
        sd_h = np.ones(hermite_order+1)
        self.theta_v_bar = np.concatenate((V_bar, sigma_bar, h_bar))
        self.sd_theta_v = np.concatenate((sd_V, sd_sigma, sd_h))

    def generate_random_params(self):
        """
        Returns a more or less plausible realization of the distribution function f
        and the Gauss-Hermite parameter theta_v
        :return: f, theta_v
        """
        f = RandomGMM_DistributionFunction(modgrid=self.op.ssps).F
        theta_v = np.random.normal(loc=self.theta_v_bar, scale=self.sd_theta_v)
        return f, theta_v

    def generate_measurement(self, f, theta_v, noise_lvl, noise="absolute"):
        """
        Generates a random measurement of G(f, theta_v).
        :param f: a distribution function (in image-format)
        :param theta_v: the gauss-hermite parameter
        :param sigma: the noise level
        :param noise: If noise="absolute", noise is simply sigma*normal(0,1).
        If noise="relative", the noise level is adapted to the scale of
        the measurement: sigma*(max(y)-min(y))*normal(0,1).
        :raises NotImplementedError: If 'noise' is neither "absolute" nor "relative".
        :return: y, y_bar, delta. y is the noisy measurement corresponding
        to the "truth" y_bar. sd is the standard deviation of the noise.
        """
        y_bar = self.op.evaluate(f, theta_v)
        # generate noise
        if noise=="absolute":
            delta = noise_lvl
        elif noise=="relative":
            delta = noise_lvl*(max(y_bar)-min(y_bar))
        else:
            raise NotImplementedError
        noi = np.random.normal(scale=delta, size=y_bar.size)
        # generate noisy measurement
        y = y_bar + noi
        return y, y_bar, delta