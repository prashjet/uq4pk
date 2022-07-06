
import numpy as np
from pathlib import Path

from ..mock_data.experiment_parameters import LMD_MIN, LMD_MAX, DV

# Parameters for visualization.
RATIO = 0.5
SIGMA_MIN = 1.
NUM_SIGMA = 10
STEPSIZE = 1.5
sigma2_list = [SIGMA_MIN + n * STEPSIZE for n in range(NUM_SIGMA)]
# Incorporate ratio.
SIGMA_LIST = [np.array([RATIO * sigma2, sigma2]) for sigma2 in sigma2_list]

SIGMA_INDEX = 1        # Index of scale for one-dimensional visualization
SLICE_INDEX = 7        # Index for one-dimensional visualization

# Parameters for blob detection
RTHRESH1 = 0.02
RTHRESH2 = RTHRESH1
OVERLAP1 = 0.5
OVERLAP2 = 0.5
REGFACTOR = 500

# Filenames
DATA = Path("src/mock_data/snr1000")
MAP = Path("blob_detection_map.npy")
LOWER_STACK = Path("blob_detection_lower_stack.npy")
UPPER_STACK = Path("blob_detection_upper_stack.npy")
EXAMPLE_MAP = Path("data/filtered_map.npy")
EXAMPLE_LOWER = Path("data/lb.npy")
EXAMPLE_UPPER = Path("data/ub.npy")
TIMES = Path("blob_detection_times.csv")