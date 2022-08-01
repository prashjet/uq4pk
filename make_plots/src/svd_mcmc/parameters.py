"""
Collection of parameters for q-test.
"""

import numpy as np
from pathlib import Path

from ..mock_data.experiment_parameters import LMD_MIN, LMD_MAX, DV          # DO NOT DELETE. THIS IS USED!


# List of values for q for which computation is done.
qlist_low = np.arange(3, 30, 3)
qlist_high = np.arange(30, 600, 100)
QLIST = np.concatenate([qlist_low, qlist_high])

# Regularization factor for model (is multiplied with SNR to give regularization parameter).
REGPARAM = 500000
NUM_BURNIN = 5000
NUM_SAMPLES = 5000

# Filenames
DATA = Path("src/mock_data/snr100")
SVDSAMPLES = Path("qtest_svdmcmc_samples.npy")
HMCSAMPLES = Path("qtest_hmc_samples.npy")
HMCCONTROL = Path("qtest_hmc_control.npy")
TIMES = Path("qtest_times.npy")
ERRORS = Path("qtest_errors.npy")