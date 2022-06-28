
"""
Create figures of filtered credible intervals.
"""


import numpy as np
from pathlib import Path

import uq4pk_src
from uq4pk_fit.inference import StatModel, MassWeightedForwardOperator
from simulate_data import load_experiment_data
from uq4pk_fit.inference.make_filter_function import make_filter_function
from src.fci.parameters import DATAFILE, FCILOW, FCIUPP, SIGMA_LIST, LMD_MIN, LMD_MAX, \
    DV, FILTERED_TRUTH, FILTERED_MAP


def compute_filtered_credible_intervals(mode: str, out: Path):
    if mode == "test":
        run_options = {"discretization": "window", "w1": 4, "w2": 4, "use_ray": True}
    elif mode == "base":
        run_options = {"discretization": "twolevel", "w1": 4, "w2": 4, "d1": 2, "d2": 2}
    else:
        run_options = {"discretization": "trivial", "use_ray": True}

    data = load_experiment_data(DATAFILE)

    # MODEL FITTING
    # Initialize model
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    forward_operator = MassWeightedForwardOperator(ssps=ssps, dv=DV)
    model = StatModel(y=data.y, y_sd=data.y_sd, forward_operator=forward_operator)
    # Fix theta at true value
    model.fix_theta_v(indices=np.arange(model.dim_theta), values=data.theta_ref)
    fitted_model = model.fit()

    # Get ground truth and MAP
    f_map = fitted_model.f_map.reshape((12, 53))
    f_true = data.f_true.flatten()

    # Compute FCI stack
    if mode in ["test", "base"]:
        low_stack, upp_stack = fitted_model.approx_fci_stack(alpha=0.05, sigma_list=SIGMA_LIST, options=run_options)
    else:
        low_stack, upp_stack = fitted_model.fci_stack(alpha=0.05, sigma_list=SIGMA_LIST)

    # Save stack
    n_scales = len(SIGMA_LIST)
    for i in range(n_scales):
        sigma_i = SIGMA_LIST[i]
        run_options["sigma"] = sigma_i
        ffunction, _, _, _ = make_filter_function(m_f=12, n_f=53, dim_theta_v=0, options={"sigma": sigma_i})
        map_t = ffunction.evaluate(f_map.flatten()).reshape((12, 53))
        truth_t = ffunction.evaluate(f_true).reshape((12, 53))
        np.save(str(out / FILTERED_MAP[i]), map_t)
        np.save(str(out / FILTERED_TRUTH[i]), truth_t)
        # Store the corresponding FCIs, then visualize them.
        fci_low = low_stack[i]
        fci_upp = upp_stack[i]
        np.save(str(out / FCILOW[i]), fci_low)
        np.save(str(out / FCIUPP[i]), fci_upp)
