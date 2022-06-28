"""
file: m54/parameters.py
"""


import numpy as np
from pathlib import Path

REGFACTOR = 5000.

THETA_V = np.array([146., 3., 1., 0., 0., 0.088, 0.187])
RATIO = 0.5
SIGMA_MIN = 1.
NUM_SIGMA = 10
STEPSIZE = 1.5
RTHRESH1 = 0.01
RTHRESH2 = 0.0
OVERLAP1 = 0.5
OVERLAP2 = 0.5
# parameters for MCMC
BURNIN_ETA_ALPHA = 5000
NSAMPLE_ETA_ALPHA = 5000
BURNIN_BETA_TILDE = 10000
NSAMPLE_BETA_TILDE = 10000

sigma2_list = [SIGMA_MIN + n * STEPSIZE for n in range(NUM_SIGMA)]
# Incorporate ratio.
SIGMA_LIST = [np.array([RATIO * sigma2, sigma2]) for sigma2 in sigma2_list]

# FILE LOCATIONS
MAP_FILE = Path("m54_map.npy")
LOWER_STACK_FILE = Path("m54_lower_stack.npy")
UPPER_STACK_FILE = Path("m54_upper_stack.npy")
MEDIAN_FILE = Path("m54_median.npy")
SAMPLE_FILE = Path("m54_samples.npy")
GROUND_TRUTH = Path("m54_ground_truth.npy")
LOWER_STACK_MCMC = Path("m54_lower_stack_mcmc.npy")
UPPER_STACK_MCMC = Path("m54_upper_stack_mcmc.npy")
LOWER_STACK_APPROX = Path("m54_lower_stack_approx.npy")
UPPER_STACK_APPROX = Path("m54_upper_stack_approx.npy")
MARGINAL = Path("m54_marginal_opt.npy")
MARGINAL_MCMC = Path("m54_marginal_mcmc.npy")
TIMES = Path("m54_computation_times.csv")
YMAP = Path("m54_y_map.npy")
YMED = Path("m54_y_med.npy")
DATA = Path("m54_data.npy")
MASK = Path("m54_mask.npy")
YSAMPLES = Path("m54_y_samples.npy")
PREDICTIVE_MCMC = Path("m54_predictive_mcmc.npy")
PREDICTIVE_OPT = Path("m54_predictive_opt.npy")

here = Path("src/m54")

MOCK1_FILE = Path("mock_data/snr1000_y.npy")
MOCK_SD1_FILE = Path("mock_data/snr1000_y_sd.npy")
MOCK2_FILE = Path("mock_data/snr100_y.npy")
MOCK_SD2_FILE = Path("mock_data/snr100_y_sd.npy")
MOCK_GT_FILE = Path("mock_data") / GROUND_TRUTH

MOCK_GT = here / MOCK_GT_FILE
MOCK1 = here / MOCK1_FILE
MOCK_SD1 = here / MOCK_SD1_FILE
MOCK2 = here / MOCK2_FILE
MOCK_SD2 = here / MOCK_SD2_FILE