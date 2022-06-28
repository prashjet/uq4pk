

from pathlib import Path
import numpy as np

from ..experiment_data.experiment_parameters import LMD_MIN, LMD_MAX, DV


scale = 5
SIGMA = np.array([0.5 * np.sqrt(2 * scale), np.sqrt(2 * scale)])
N1 = 10
N2 = 50

CLIST = [4, 6, 8, 10, 12, 12]
DLIST = [8, 12, 16, 20, 24, 36]
D1 = 2
D2 = 4
W1LIST = [2, 3, 4, 5, 6, 6]
W2LIST = [2, 3, 4, 5, 6, 9]

# Filenames
DATAFILE = Path("src/experiment_data/snr100")
ERRORS_WINDOW_FILE = Path(f"errors_window.npy")
ERRORS_TWOLEVEL_FILE = Path(f"errors_twolevel.npy")
HEURISTIC_WINDOW_FILE1 = Path(f"heuristic_window_{N1}.npy")
HEURISTIC_WINDOW_FILE2 = Path(f"heuristic_window_{N2}.npy")
HEURISTIC_TWOLEVEL_FILE1 = Path(f"heuristic_twolevel_{N1}.npy")
HEURISTIC_TWOLEVEL_FILE2 = Path(f"heuristic_twolevel_{N2}.npy")