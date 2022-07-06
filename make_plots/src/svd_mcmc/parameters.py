"""
Collection of parameters for q-test.
"""

import numpy as np
from pathlib import Path

from ..mock_data.experiment_parameters import LMD_MIN, LMD_MAX, DV          # DO NOT DELETE. THIS IS USED!


# List of values for q for which computation is done.
QLIST = np.arange(4, 60, 4)

# Regularization factor for model (is multiplied with SNR to give regularization parameter).
REGFACTOR = 5000
NUM_BURNIN = 10000
NUM_SAMPLES = 10000

# Filenames
DATA = Path("src/mock_data/snr100")
SVDSAMPLES = Path("qtest_svdmcmc_samples.npy")
HMCSAMPLES = Path("qtest_hmc_samples.npy")
TIMES = Path("qtest_times.npy")
DIVERGENCES = Path("qtest_divergence.npy")