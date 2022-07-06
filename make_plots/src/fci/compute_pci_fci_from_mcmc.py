
import numpy as np
from pathlib import Path

from uq4pk_fit.inference.fcis_from_samples2d import fcis_from_samples2d, pcis_from_samples2d
from uq4pk_fit.inference.make_filter_function import make_filter_function
from .parameters import SAMPLES, PCILOW, FCILOW, PCIUPP, FCIUPP, SIGMA_LIST

def compute_pci_fci_from_mcmc(mode: str, out: Path):

    samples = np.load(str(out / SAMPLES))

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
        ffunction, _, _, _ = make_filter_function(m_f=12, n_f=53, dim_theta_v=0, options={"sigma": sigma_i})
        #map_t = ffunction.evaluate(f_map.flatten()).reshape((12, 53))
        #truth_t = ffunction.evaluate(f_true).reshape((12, 53))
        #np.save(str(out / FILTERED_MAP[i]), map_t)
        #np.save(str(out / FILTERED_TRUTH[i]), truth_t)
        # Store the corresponding FCIs, then visualize them.
        fci_low_i = fci_low[i].reshape(12, 53)
        fci_upp_i = fci_upp[i].reshape(12, 53)
        np.save(str(out / FCILOW[i]), fci_low_i)
        np.save(str(out / FCIUPP[i]), fci_upp_i)