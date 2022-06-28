"""

"""


import numpy as np
from pathlib import Path

from uq4pk_fit.inference import fcis_from_samples2d, marginal_ci_from_samples
from uq4pk_fit.uq_mode import credible_intervals
from src.m54.parameters import LOWER_STACK_MCMC, UPPER_STACK_MCMC, SAMPLE_FILE, SIGMA_LIST, MARGINAL_MCMC, YSAMPLES, \
    PREDICTIVE_MCMC


def m54_samples_to_fcis(out: Path):
    """
    Creates FCIs from samples
    """

    # Load samples.
    samples = np.load(str(out / SAMPLE_FILE))

    # Compute FCIs from shaped samples.
    lower_stack, upper_stack = fcis_from_samples2d(alpha=0.05, samples=samples, sigmas=SIGMA_LIST)
    # Also compute age-marginal
    age_lb, age_ub = marginal_ci_from_samples(alpha=0.05, axis=0, samples=samples)
    age_marginal = np.row_stack([age_ub, age_lb])
    # Also compute posterior predictive intervals.
    y_samples = np.load(str(out / YSAMPLES))
    lb_y, ub_y = credible_intervals(samples=y_samples, alpha=0.05)
    ci_y = np.row_stack([ub_y, lb_y])

    # Store FCIs and marginal CIs.
    np.save(str(out / LOWER_STACK_MCMC), lower_stack)
    np.save(str(out / UPPER_STACK_MCMC), upper_stack)
    np.save(str(out / MARGINAL_MCMC), age_marginal)
    np.save(str(out / PREDICTIVE_MCMC), ci_y)