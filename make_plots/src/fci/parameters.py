

from math import sqrt
import numpy as np
from pathlib import Path

from src.experiment_data.experiment_parameters import LMD_MIN, LMD_MAX, DV


SCALES = [1., 5., 20.]
ratio = 0.5
# Make SIGMA_LIST
sigma2_list = [sqrt(2 * t) for t in SCALES]
# Incorporate ratio.
SIGMA_LIST = [np.array([ratio * sigma2, sigma2]) for sigma2 in sigma2_list]

# FILE LOCATIONS
DATAFILE = Path("src/experiment_data/snr2000")
MAPFILE = Path("pci_map")
GROUND_TRUTH = Path("pci_ground_truth")
PCILOW = Path("pci_low")
PCIUPP = Path("pci_upp")

# Make list of names for filtered intervals / images at different scales.
FCILOW = [Path(f"fci_low_{int(t)}") for t in SCALES]
FCIUPP = [Path(f"fci_upp_{int(t)}") for t in SCALES]
FILTERED_MAP = [Path(f"fci_map_{int(t)}") for t in SCALES]
FILTERED_TRUTH = [Path(f"fci_ground_truth_{int(t)}") for t in SCALES]

assert len(FCILOW) == len(SCALES)
assert len(FCIUPP) == len(SCALES)
assert len(FILTERED_MAP) == len(SCALES)
assert len(FILTERED_TRUTH) == len(SCALES)