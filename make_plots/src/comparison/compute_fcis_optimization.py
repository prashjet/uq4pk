"""
Computes the FCI stack necessary used blob detection.
"""


import numpy as np
from pathlib import Path

from uq4pk_fit.inference import LightWeightedForwardOperator, StatModel
import uq4pk_src
from ..mock import ExperimentData
from .parameters import LMD_MAX, LMD_MIN, DV, SIGMA_LIST, MAPFILE, TRUTHFILE, LOWER_STACK_OPT, \
    UPPER_STACK_OPT, REGFACTOR


def compute_fcis_optimization(mode: str, data: ExperimentData, out: Path):
    if mode == "test":
        run_options = {"discretization": "window", "w1": 3, "w2": 3, "use_ray": True}
    elif mode == "base":
        run_options = {"discretization": "trivial", "use_ray": True, "optimizer": "SCS", "a": 2, "b": 2}
    else:
        run_options = {"eps": 1e-4}

    # Set up SSPS grid and forward operator
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    forward_operator = LightWeightedForwardOperator(ssps=ssps, dv=DV, theta=data.theta_ref)

    # MODEL SETUP
    y = data.y
    y_sd = data.y_sd
    model = StatModel(y=y, y_sd=y_sd, forward_operator=forward_operator)
    model.fix_theta_v(indices=np.arange(data.theta_ref.size), values=data.theta_ref)
    snr = np.linalg.norm(y) / np.linalg.norm(y_sd)
    model.beta1 = REGFACTOR * snr

    # MODEL FITTING
    fitted_model = model.fit()
    f_map = fitted_model.f_map
    f_true = data.f_ref
    print(f"SNR = {snr}.")
    print(f"Scale = {np.sum(y)}")
    print(f"sum(f_map) = {np.sum(f_map)}, max(f_map) = {f_map.max()}")
    print(f"sum(f_true) = {np.sum(f_true)}, max(f_true) = {f_true.max()}")

    # COMPUTE FCI STACK
    if mode in ["test", "base"]:
        lower_stack, upper_stack = fitted_model.approx_fci_stack(alpha=0.05, sigma_list=SIGMA_LIST, options=run_options)
    else:
        lower_stack, upper_stack = fitted_model.fci_stack(alpha=0.05, sigma_list=SIGMA_LIST, options=run_options)
    # Save everything
    np.save(arr=f_true, file=str(out / TRUTHFILE))
    np.save(arr=f_map, file=str(out / MAPFILE))
    np.save(file=str(out / LOWER_STACK_OPT), arr=lower_stack)
    np.save(file=str(out / UPPER_STACK_OPT), arr=upper_stack)
