

import numpy as np
from pathlib import Path

from uq4pk_fit.inference import fcis_from_samples2d, marginal_ci_from_samples
from uq4pk_fit.uq_mode import credible_intervals
from .parameters import LOWER_STACK_SVDMCMC, LOWER_STACK_HMC, UPPER_STACK_SVDMCMC, UPPER_STACK_HMC, \
    SAMPLES_SVDMCMC, SAMPLES_HMC, SIGMA_LIST, MARGINAL_SVDMCMC, YSAMPLES_SVDMCMC, \
    PREDICTIVE_SVDMCMC, MARGINAL_HMC, YSAMPLES_HMC, PREDICTIVE_HMC


def m54_samples_to_fcis(out: Path, sampling: str):
    """
    Creates FCIs from samples
    """
    if sampling == "svdmcmc":
        sample_file = SAMPLES_SVDMCMC
        lower_stack_file = LOWER_STACK_SVDMCMC
        upper_stack_file = UPPER_STACK_SVDMCMC
        marginal_file = MARGINAL_SVDMCMC
        ysamples_file = YSAMPLES_SVDMCMC
        predictive_file = PREDICTIVE_SVDMCMC
    elif sampling == "hmc":
        sample_file = SAMPLES_HMC
        lower_stack_file = LOWER_STACK_HMC
        upper_stack_file = UPPER_STACK_HMC
        marginal_file = MARGINAL_HMC
        ysamples_file = YSAMPLES_HMC
        predictive_file = PREDICTIVE_HMC
    else:
        raise NotImplementedError("Unknown sampler.")
    # Load samples.
    samples = np.load(str(out / sample_file))

    # Compute FCIs from shaped samples.
    lower_stack, upper_stack = fcis_from_samples2d(alpha=0.05, samples=samples, sigmas=SIGMA_LIST)
    # Also compute age-marginal
    age_lb, age_ub = marginal_ci_from_samples(alpha=0.05, axis=0, samples=samples)
    age_marginal = np.row_stack([age_ub, age_lb])
    # Also compute posterior predictive intervals.
    y_samples = np.load(str(out / ysamples_file))
    lb_y, ub_y = credible_intervals(samples=y_samples, alpha=0.05)
    ci_y = np.row_stack([ub_y, lb_y])

    # Store FCIs and marginal CIs.
    np.save(str(out / lower_stack_file), lower_stack)
    np.save(str(out / upper_stack_file), upper_stack)
    np.save(str(out / marginal_file), age_marginal)
    np.save(str(out / predictive_file), ci_y)