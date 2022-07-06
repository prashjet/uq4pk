
import numpy as np
import pandas
from pathlib import Path
from time import time

import uq4pk_src
from .m54_mcmc_sample import m54_mcmc_sample
from .m54_samples_to_fcis import m54_samples_to_fcis
from .parameters import TIMES, MOCK1, MOCK_SD1, MOCK2, MOCK_SD2, MOCK_GT, GROUND_TRUTH, DATA, PPXF

mock1_name = "m54_mock1000"
mock2_name = "m54_mock100"
real_name = "m54_real"


def compute_m54(mode: str, out: Path):
    mock1 = out / mock1_name
    mock2 = out / mock2_name
    real = out / real_name
    #_compute_mock_data(mode=mode, out=mock1, mock_y=MOCK1, mock_sd=MOCK_SD1, mock_f=MOCK_GT)
    #_compute_mock_data(mode=mode, out=mock2, mock_y=MOCK2, mock_sd=MOCK_SD2, mock_f=MOCK_GT)
    _compute_real_data(mode=mode, out=real)


def _compute_mock_data(mode: str, out: Path, mock_y: Path, mock_sd: Path, mock_f: Path):
    # Get mock data and store.
    y_mock = np.load(str(mock_y))
    y_mock_sd = np.load(str(mock_sd))
    np.save(str(out / DATA), y_mock)
    # Get ground truth for mock data and store it.
    ground_truth = np.load(str(mock_f))
    np.save(str(out / GROUND_TRUTH), ground_truth)
    # First, compute FCIs via sampling.
    m54_mcmc_sample(mode=mode, out=out, y=y_mock, y_sd=y_mock_sd, sampling="svdmcmc")
    m54_samples_to_fcis(out=out, sampling="svdmcmc")
    # Finally, compute FCIs via full HMC.
    m54_mcmc_sample(mode=mode, out=out, y=y_mock, y_sd=y_mock_sd, sampling="hmc")
    m54_samples_to_fcis(out=out, sampling="hmc")


def _compute_real_data(mode: str, out: Path):
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
    # First, compute FCIs via SVD-MCMC.
    t0 = time()
    m54_mcmc_sample(mode=mode, out=out, y=y_real, y_sd=y_sd_real, sampling="svdmcmc")
    m54_samples_to_fcis(out=out, sampling="svdmcmc")
    t1 = time()
    time_svdmcmc = t1 - t0
    print(f"---------- SVD-MCMC TOOK {time_svdmcmc} seconds.")
    # Then, compute with full HMC.
    m54_mcmc_sample(mode=mode, out=out, y=y_real, y_sd=y_sd_real, sampling="hmc")
    m54_samples_to_fcis(out=out, sampling="hmc")
    t2 = time()
    time_hmc = t2 - t1
    print(f"---------- FULL HMC TOOK {time_hmc} seconds.")
    times = np.array([time_svdmcmc, time_hmc]).reshape(1, 2)
    times_frame = pandas.DataFrame(data=times, columns=["SVDMCMC", "HMC"])
    times_frame.to_csv(out / TIMES)