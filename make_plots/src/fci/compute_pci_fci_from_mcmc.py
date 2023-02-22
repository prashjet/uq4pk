
import numpy as np
from pathlib import Path

from uq4pk_fit.filtered_credible_intervals.fcis_from_samples2d import fcis_from_samples2d, pcis_from_samples2d
from uq4pk_fit.filtered_credible_intervals.filter import BesselFilterFunction2D
from .parameters import SAMPLES, PCILOW, FCILOW, PCIUPP, FCIUPP, SIGMA_LIST, MAPFILE, GROUND_TRUTH, FILTERED_TRUTH, \
    FILTERED_MAP

def compute_pci_fci_from_mcmc(mode: str, out: Path):
    """
    Computes pixel-wise credible intervals (PCIs) and filtered credible intervals from the pre-computed MCMC samples
    and stores them in .npy-files.
    """
    # Load samples
    samples = np.load(str(out / SAMPLES))
    # Load MAP and ground truth.
    f_map = np.load(str(out / MAPFILE))
    f_true = np.load(str(out / GROUND_TRUTH))

    # COMPUTE FCIs from samples.
    alpha = 0.05
    fci_low, fci_upp = fcis_from_samples2d(alpha=alpha, samples=samples, sigmas=SIGMA_LIST)
    pci_low, pci_upp = pcis_from_samples2d(alpha=alpha, samples=samples)

    np.save(arr=pci_low.reshape(12, 53), file=str(out / PCILOW))
    np.save(arr=pci_upp.reshape(12, 53), file=str(out / PCIUPP))
    # Save stack
    n_scales = len(SIGMA_LIST)
    for i in range(n_scales):
        sigma_i = SIGMA_LIST[i]
        filter = BesselFilterFunction2D(m=12, n=53, sigma=sigma_i)
        map_t = filter.evaluate(f_map.flatten()).reshape((12, 53))
        truth_t = filter.evaluate(f_true).reshape((12, 53))
        np.save(str(out / FILTERED_MAP[i]), map_t)
        np.save(str(out / FILTERED_TRUTH[i]), truth_t)
        # Store the corresponding FCIs, then visualize them.
        fci_low_i = fci_low[i].reshape(12, 53)
        fci_upp_i = fci_upp[i].reshape(12, 53)
        np.save(str(out / FCILOW[i]), fci_low_i)
        np.save(str(out / FCIUPP[i]), fci_upp_i)