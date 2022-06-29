"""
Compares FCIs computed from pseudosamples to FCIs computed with optimization, to find out what's going wrong.
"""


import numpy as np
from pathlib import Path

from uq4pk_fit.inference import fcis_from_samples2d
from ..mock import ExperimentData
from ..util.geometric_median import geometric_median
from .parameters import SIGMA_LIST, SAMPLEFILE, LOWER_STACK_MCMC, \
    UPPER_STACK_MCMC, MEDIANFILE


def compute_fcis_mcmc(data: ExperimentData, out: Path):
    # Load samples.
    samples = np.load(str(out / SAMPLEFILE))

    # COMPUTE FCIs from samples.
    alpha = 0.05
    lower_stack, upper_stack = fcis_from_samples2d(alpha=alpha, samples=samples, sigmas=SIGMA_LIST)
    # Also compute geometric median.
    f_med = geometric_median(samples)

    np.save(arr=lower_stack, file=str(out / LOWER_STACK_MCMC))
    np.save(arr=upper_stack, file=str(out / UPPER_STACK_MCMC))
    np.save(arr=f_med, file=str(out / MEDIANFILE))
