
from pathlib import Path

import uq4pk_src
from ..mock import simulate, get_f, save_experiment_data
from .experiment_parameters import THETA_V, LMD_MAX, LMD_MIN, DV, FUNCTION_NO


TESTFUNCTION_DIR = Path("test_functions")


def make_experiment_data(name: str, snr: float, function_number: int):
    # Create ssps-grid.
    ssps = uq4pk_src.model_grids.MilesSSP(lmd_min=LMD_MIN, lmd_max=LMD_MAX)
    # Create ExperimentData.
    f_im = get_f(location=str("test_functions"), numbers=[function_number])
    experiment_data = simulate(name=name, f_im=f_im, snr=snr, theta_v=THETA_V, light_weighted=True, dv=DV, ssps=ssps)
    # Store ExperimentData.
    save_experiment_data(experiment_data, savename=name)


make_experiment_data(name="snr10", snr=10, function_number=FUNCTION_NO)
make_experiment_data(name="snr100", snr=100, function_number=FUNCTION_NO)
make_experiment_data(name="snr500", snr=500, function_number=FUNCTION_NO)
make_experiment_data(name="snr1000", snr=1000, function_number=FUNCTION_NO)
make_experiment_data(name="snr2000", snr=2000, function_number=FUNCTION_NO)