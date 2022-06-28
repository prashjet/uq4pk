"""
file: computational_aspects/computational_aspects_compute.py
"""


import numpy as np
from pathlib import Path

from simulate_data import load_experiment_data
import uq4pk_src
from uq4pk_fit.inference import StatModel, LightWeightedForwardOperator
from src.computational_aspects.parameters import DATAFILE, SIGMA, N1, N2, CLIST, DLIST, D1, D2, W1LIST, \
    W2LIST, ERRORS_WINDOW_FILE, ERRORS_TWOLEVEL_FILE, HEURISTIC_TWOLEVEL_FILE1, HEURISTIC_TWOLEVEL_FILE2, \
    HEURISTIC_WINDOW_FILE1, HEURISTIC_WINDOW_FILE2, LMD_MIN, LMD_MAX, DV

NTEST = 10

def compute_localization_plots(mode: str, out: Path):
    # Reset parameters if in test_mode.
    if mode == "test":
        n1 = 5
        n2 = 10
        c_list = [4, 8, 12]
        d_list = [8, 16, 36]
        d1 = 2
        d2 = 4
        w1_list = [2, 4, 6]
        w2_list = [2, 4, 9]
    else:
        n1 = N1
        n2 = N2
        c_list = CLIST
        d_list = DLIST
        d1 = D1
        d2 = D2
        w1_list = W1LIST
        w2_list = W2LIST


    data = load_experiment_data(DATAFILE)

    # MODEL SETUP
    # Initialize model
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    forward_operator = LightWeightedForwardOperator(ssps=ssps, dv=DV, theta=data.theta_ref)
    y = data.y
    y_sd = data.y_sd
    model = StatModel(y=y, y_sd=y_sd, forward_operator=forward_operator)
    # Fix theta at true value
    model.fix_theta_v(indices=np.arange(model.dim_theta), values=data.theta_ref)
    fitted_model = model.fit()

    for n, window_file, twolevel_file in zip([n1, n2], [HEURISTIC_WINDOW_FILE1, HEURISTIC_WINDOW_FILE2],
                                             [HEURISTIC_TWOLEVEL_FILE1, HEURISTIC_TWOLEVEL_FILE2]):
        # Apply heuristic
        times1, errors1 = fitted_model.make_localization_plot(alpha=0.05, n_sample=n, sigma=SIGMA, w1_list=c_list,
                                                              w2_list=d_list, discretization_name="window")
        times_errors1 = np.row_stack([times1, errors1])
        # Save as .npy file.
        np.save(file=str(out / window_file), arr=times_errors1)

        times2, errors2 = fitted_model.make_localization_plot(alpha=0.05, n_sample=n, sigma=SIGMA, w1_list=w1_list,
                                                              w2_list=w2_list, discretization_name="twolevel", d1=d1,
                                                              d2=d2)
        times_errors2 = np.row_stack([times2, errors2])
        np.save(file=str(out / twolevel_file), arr=times_errors2)

    # Now, make exact localization plots (this will take some time).
    if mode == "test":
        times1, errors1 = fitted_model.make_localization_plot(alpha=0.05, n_sample=NTEST, sigma=SIGMA, w1_list=c_list,
                                                              w2_list=d_list, discretization_name="window")
        times2, errors2 = fitted_model.make_localization_plot(alpha=0.05, n_sample=NTEST, sigma=SIGMA, w1_list=w1_list,
                                                              w2_list=w2_list, discretization_name="twolevel", d1=d1,
                                                              d2=d2)
    else:
        times1, errors1 = fitted_model.make_localization_plot(alpha=0.05, sigma=SIGMA, w1_list=c_list, w2_list=d_list,
                                                          discretization_name="window")
        times2, errors2 = fitted_model.make_localization_plot(alpha=0.05, sigma=SIGMA, w1_list=w1_list, w2_list=w2_list,
                                                          discretization_name="twolevel", d1=d1, d2=d2)

    times_errors1 = np.row_stack([times1, errors1])
    times_errors2 = np.row_stack([times2, errors2])

    np.save(file=str(out / ERRORS_WINDOW_FILE), arr=times_errors1)
    np.save(file=str(out / ERRORS_TWOLEVEL_FILE), arr=times_errors2)
