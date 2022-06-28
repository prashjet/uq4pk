"""
file: m54_compute_cvxopt.py
"""

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from src.m54.parameters import SIGMA_LIST, MAP_FILE, LOWER_STACK_FILE, UPPER_STACK_FILE, MARGINAL, THETA_V, YMAP, MASK,\
    PREDICTIVE_OPT
from .m54_fit_model import m54_fit_model


def m54_compute_cvxopt(mode: str, out: Path, y: np.ndarray, y_sd: np.ndarray):

    # IF TEST MODE, USE REDUCED SETTINGS
    if mode == "test":
        run_options = {"discretization": "window", "w1": 5, "w2": 5, "use_ray": True}
        d = 100
        eps = 1e-2
    elif mode == "base":
        run_options = {"discretization": "trivial", "optimizer": "SCS", "a": 2, "b": 2}
        d = 3
        eps = 1e-3
    else:
        run_options = {"eps": 1e-3}
        d = 1
        eps = 1e-4

    # ------------------------------------ FIT MODEL TO DATA

    m54_model = m54_fit_model(y=y, y_sd=y_sd, theta_v=THETA_V)
    fitted_model = m54_model.fitted_model
    # Get MAP estimate
    f_map = fitted_model.f_map.clip(min=0.)
    # Get posterior predictive MAP
    y_map = fitted_model.y_map
    # Compare scale of y_map with scale of y
    scale = fitted_model.scale
    map_scale = np.sum(y_map)
    print(f"Scale of y_map is {map_scale} while scale of y is {scale}. Ratio: {map_scale / scale}.")
    np.save(str(out / YMAP), y_map)

    # ------------------------------------ COMPUTE CREDIBLE INTERVALS
    # Compute FCIs.
    if mode in ["test", "base"]:
        lower_stack, upper_stack = fitted_model.approx_fci_stack(alpha=0.05, sigma_list=SIGMA_LIST, options=run_options)
    else:
        lower_stack, upper_stack = fitted_model.fci_stack(alpha=0.05, sigma_list=SIGMA_LIST, options=run_options)
    # Compute age-marginal simultaneous CI.
    age_lb, age_ub = fitted_model.marginal_credible_intervals(alpha=0.05, axis=0)
    age_ci = np.row_stack([age_ub, age_lb])
    # Compute posterior predictive credible intervals.
    y_lb, y_ub = fitted_model.posterior_predictive_credible_intervals(alpha=0.05, eps=eps, d=d)
    y_ci = np.row_stack([y_ub, y_lb])
    # Sanity check.
    error_vec = np.row_stack([(y_lb - y_map).clip(min=0.), (y_map - y_ub).clip(min=0.)])
    downsampling_error = np.max(error_vec) / np.max(y_map)
    print(f"Relative downsampling error: {downsampling_error}.")

    np.save(str(out / LOWER_STACK_FILE), lower_stack)
    np.save(str(out / UPPER_STACK_FILE), upper_stack)
    np.save(str(out / MAP_FILE), f_map)
    np.save(str(out / MARGINAL), age_ci)
    np.save(str(out / PREDICTIVE_OPT), y_ci)

    # Also store mask
    mask = m54_model.forward_operator.mask
    np.save(str(out / MASK), mask)