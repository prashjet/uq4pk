"""
This file sets values for default model parameters.
"""

import numpy as np


# DEFAULT PARAMETERS FOR EXPERIMENT SETUP
HIGH_SNR = 3000
LOW_SNR = 100

GUESS_THETA = np.array([30, 100, 1., 0., 0., -0.05, 0.1])
STDEV_THETA = np.array([30., 100., 1., .05, .05, .05, .1])
NOISE_THETA = 0.05      # default relative error of the guess wrt the true value of theta.

