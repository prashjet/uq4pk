
import numpy as np
from pathlib import Path

REGFACTORS = [1000., 500000.]

RATIO = 0.5
SIGMA_MIN = 1.
NUM_SIGMA = 10
STEPSIZE = 1.5
RTHRESH1 = 0.02
RTHRESH2 = 0.02
OVERLAP1 = 0.5
OVERLAP2 = 0.5
# parameters for SVD-MCMC
SVDMCMC_BURNIN = 5000
SVDMCMC_NSAMPLES = 10000
# parameters for HMC
HMC_BURNIN = 5000
HMC_NSAMPLES = 10000

sigma2_list = [SIGMA_MIN + n * STEPSIZE for n in range(NUM_SIGMA)]
# Incorporate ratio.
SIGMA_LIST = [np.array([RATIO * sigma2, sigma2]) for sigma2 in sigma2_list]

# FILE LOCATIONS
MOCK1_NAME = "m54_mock1000"
MOCK2_NAME = "m54_mock100"
REAL1_NAME = "m54_lowreg"
REAL2_NAME = "m54_highreg"

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
AGE_SVDMCMC = Path("m54_age_svdmcmc.npy")
AGE_HMC = Path("m54_age_hmc.npy")
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