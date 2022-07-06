"""
file: m54/parameters.py
"""


import numpy as np
from pathlib import Path

REGFACTOR = 50.  #5000.

THETA_V = np.array([146., 3., 1., 0., 0., 0.014, 0.169])
RATIO = 0.5
SIGMA_MIN = 1.
NUM_SIGMA = 10
STEPSIZE = 1.5
RTHRESH1 = 0.01
RTHRESH2 = 0.01
OVERLAP1 = 0.5
OVERLAP2 = 0.5
# parameters for SVD-MCMC
SVDMCMC_BURNIN = 10000
SVDMCMC_NSAMPLES = 10000
# parameters for HMC
HMC_BURNIN = 10000
HMC_NSAMPLES = 10000

sigma2_list = [SIGMA_MIN + n * STEPSIZE for n in range(NUM_SIGMA)]
# Incorporate ratio.
SIGMA_LIST = [np.array([RATIO * sigma2, sigma2]) for sigma2 in sigma2_list]

# FILE LOCATIONS
MAP_FILE = Path("m54_map.npy")
LOWER_STACK_FILE = Path("m54_lower_stack.npy")
UPPER_STACK_FILE = Path("m54_upper_stack.npy")
MEAN_SVDMCMC = Path("m54_mean_svdmcmc.npy")
MEAN_HMC = Path("m54_mean_hmc.npy")
SAMPLES_SVDMCMC = Path("m54_samples_svdmcmc.npy")
SAMPLES_HMC = Path("m54_samples_hmc.npy")
GROUND_TRUTH = Path("m54_ground_truth.npy")
PPXF = Path("m54_ppxf.npy")
LOWER_STACK_SVDMCMC = Path("m54_lower_stack_svdmcmc.npy")
UPPER_STACK_SVDMCMC = Path("m54_upper_stack_svdmcmc.npy")
LOWER_STACK_OPT = Path("m54_lower_stack_approx.npy")
UPPER_STACK_OPT = Path("m54_upper_stack_approx.npy")
LOWER_STACK_HMC = Path("m54_lower_stack_hmc.npy")
UPPER_STACK_HMC = Path("m54_upper_stack_hmc.npy")
MARGINAL_OPT = Path("m54_marginal_opt.npy")
MARGINAL_SVDMCMC = Path("m54_marginal_svdmcmc.npy")
MARGINAL_HMC = Path("m54_marginal_hmc.npy")
TIMES = Path("m54_computation_times.csv")
YMAP = Path("m54_y_map.npy")
YMEAN_SVDMCMC = Path("m54_y_mean_svdmcmc.npy")
YMEAN_HMC = Path("m54_y_mean_hmc.npy")
DATA = Path("m54_data.npy")
MASK = Path("m54_mask.npy")
YSAMPLES_SVDMCMC = Path("m54_y_samples_svdmcmc.npy")
YSAMPLES_HMC = Path("m54_y_samples_hmc.npy")
PREDICTIVE_SVDMCMC = Path("m54_predictive_svdmcmc.npy")
PREDICTIVE_OPT = Path("m54_predictive_opt.npy")
PREDICTIVE_HMC = Path("m54_predicitive_hmc.npy")

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