
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

# Run options for speedup. Note that these options lead to a reduction factor >= 10 in overall computation time on
# my PC, but YMMV depending on the parameter settings and used hardware.
SPEEDUP_OPTIONS = {"discretization": "twolevel", "d1": 2, "d2": 4, "w1": 5, "w2": 5, "optimizer": "SCS", "a": 2,
                   "b": 2}
# Filenames
DATA = Path("src/mock_data/snr1000")
MAP = Path("blob_detection_map.npy")
LOWER_STACK = Path("blob_detection_lower_stack.npy")
UPPER_STACK = Path("blob_detection_upper_stack.npy")
LOWER_STACK_SPEEDUP = Path("blob_detection_lower_stack_speedup.npy")
UPPER_STACK_SPEEDUP = Path("blob_detection_upper_stack_speedup.npy")
EXAMPLE_MAP = Path("data/filtered_map.npy")
EXAMPLE_LOWER = Path("data/lb.npy")
EXAMPLE_UPPER = Path("data/ub.npy")
TIMES = Path("blob_detection_times.csv")