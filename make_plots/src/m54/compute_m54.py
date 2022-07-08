
import numpy as np
import pandas
from pathlib import Path

import uq4pk_src
from .m54_fit_model import m54_fit_model
from .m54_mcmc_sample import m54_mcmc_sample
from .m54_samples_to_fcis import m54_samples_to_fcis
from .parameters import TIMES, GROUND_TRUTH, DATA, PPXF, REGFACTORS, REAL1_NAME, REAL2_NAME, MAP_FILE


def compute_m54(mode: str, out: Path):
    real1 = out / REAL1_NAME
    real2 = out / REAL2_NAME

    for out, regparam in zip([real1, real2], REGFACTORS):
        _compute_real_data(mode=mode, out=out, regparam=regparam)


def _compute_real_data(mode: str, out: Path, regparam: float):
    # Get real data and store.
    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)
    y_real = m54_data.y
    y_sd_real = m54_data.noise_level
    np.save(str(out / DATA), y_real)
    # Write out ground truth.
    ground_truth = m54_data.ground_truth.T
    np.save(str(out / GROUND_TRUTH), ground_truth)
    # Write out ppxf solution.
    ppxf = m54_data.ppxf_map_solution.T
    np.save(str(out / PPXF), ppxf)
    # First, compute and store the MAP.
    m54_model = m54_fit_model(y_real, y_sd_real, regparam=regparam)
    f_map = m54_model.fitted_model.f_map
    np.save(str(out / MAP_FILE), f_map)
    # First, compute FCIs via SVD-MCMC.
    time_svdmcmc = m54_mcmc_sample(mode=mode, out=out, y=y_real, y_sd=y_sd_real, sampling="svdmcmc", regparam=regparam)
    m54_samples_to_fcis(out=out, sampling="svdmcmc")
    print(f"---------- SVD-MCMC TOOK {time_svdmcmc} seconds.")
    # Then, compute with full HMC.
    time_hmc = m54_mcmc_sample(mode=mode, out=out, y=y_real, y_sd=y_sd_real, sampling="hmc", regparam=regparam)
    print(f"---------- FULL HMC TOOK {time_hmc} seconds.")
    m54_samples_to_fcis(out=out, sampling="hmc")
    times = np.array([time_svdmcmc, time_hmc]).reshape(1, 2)
    times_frame = pandas.DataFrame(data=times, columns=["SVDMCMC", "HMC"])
    times_frame.to_csv(out / TIMES)