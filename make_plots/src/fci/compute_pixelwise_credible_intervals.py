"""
Creates pictures of credible intervals.
"""


import numpy as np
from pathlib import Path

from uq4pk_fit.inference import StatModel, MassWeightedForwardOperator
import uq4pk_src
from ..mock import load_experiment_data
from .parameters import DATAFILE, MAPFILE, GROUND_TRUTH, PCILOW, PCIUPP, LMD_MAX, LMD_MIN, DV


def compute_pixelwise_credible_intervals(mode: str, out: Path):
    if mode == "test":
        run_options = {"discretization": "window", "w1": 4, "w2": 4, "use_ray": True}
    elif mode == "base":
        run_options = {"discretization": "twolevel", "w1": 4, "w2": 4, "d1": 2, "d2": 2, "use_ray": True}
    else:
        run_options = {"discretization": "trivial", "use_ray": True, "optimizer": "SCS"}
    data = load_experiment_data(DATAFILE)

    # MODEL SETUP
    # Initialize model
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    forward_operator = MassWeightedForwardOperator(ssps=ssps, dv=DV)
    model = StatModel(y=data.y, y_sd=data.y_sd, forward_operator=forward_operator)
    # Fix theta at true value
    model.fix_theta_v(indices=np.arange(model.dim_theta), values=data.theta_ref)
    # Fit the model by computing MAP.
    fitted_model = model.fit()

    f_map = fitted_model.f_map.reshape((12, 53))
    f_true = data.f_true.reshape((12, 53))

    # Compute PCI.
    pci_low, pci_upp = fitted_model.pci(alpha=0.05, options=run_options)

    # Store the corresponding PCIs.
    np.save(str(out / MAPFILE), f_map)
    np.save(str(out / GROUND_TRUTH), f_true)
    np.save(str(out / PCILOW), pci_low)
    np.save(str(out / PCIUPP), pci_upp)
