"""
file: comparison/parameters.py
    Here, the numeric parameters for the MCMC comparison are defined.
"""


import numpy as np
from pathlib import Path

from ..mock_data.experiment_parameters import LMD_MIN, LMD_MAX, DV


# Discretize the scale line.
RATIO = 0.5
SIGMA_MIN = 1.
NUM_SIGMA = 10
STEPSIZE = 2.
sigma2_list = [SIGMA_MIN + n * STEPSIZE for n in range(NUM_SIGMA)]
SIGMA_LIST = [np.array([RATIO * sigma2, sigma2]) for sigma2 in sigma2_list]

# Parameters for the data simulation.
SNR1 = 1000
SNR2 = 100
SNR3 = 10
NAME1 = f"snr{SNR1}"
NAME2 = f"snr{SNR2}"
NAME3 = f"snr{SNR3}"

REGFACTOR = 5000

# Parameters for MCMC
BURNIN_ETA_ALPHA = 10000
NSAMPLE_ETA_ALPHA = 10000
BURNIN_BETA_TILDE = 5000
NSAMPLE_BETA_TILDE = 5000
Q = 15
SIGMA_ALPHA = 0.1
SIGMA_ETA = 0.1

# Parameters for blob detection
RTHRESH1 = 0.1
RTHRESH2 = 0.0
OVERLAP1 = 0.5
OVERLAP2 = 0.5

# Filenames.
OUT1 = Path(NAME1)
OUT2 = Path(NAME2)
OUT3 = Path(NAME3)
DATAFILE1 = Path("src/experiment_data") / f"snr{SNR1}"
DATAFILE2 = Path("src/experiment_data") / f"snr{SNR2}"
DATAFILE3 = Path("src/experiment_data") / f"snr{SNR3}"
SAMPLEFILE = "comparison_samples.npy"
MEDIANFILE = "comparison_median.npy"
MAPFILE = "comparison_map.npy"
TRUTHFILE = "comparison_truth.npy"
LOWER_STACK_OPT = "comparison_lower_stack_opt.npy"
UPPER_STACK_OPT = "comparison_upper_stack_opt.npy"
LOWER_STACK_MCMC = "comparison_lower_stack_mcmc.npy"
UPPER_STACK_MCMC = "comparison_upper_stack_mcmc.npy"
