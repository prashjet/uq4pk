import numpy as np
from pathlib import Path

import uq4pk_src
from simulate_data import get_f
from src.m54.m54_fit_model import m54_setup_operator, m54_fit_model
from src.m54.parameters import THETA_V, MOCK1_FILE, MOCK_SD1_FILE, MOCK2_FILE, MOCK_SD2_FILE, MOCK_GT_FILE
from uq4pk_fit.visualization import plot_distribution_function


def make_test_function():
    f_im = get_f(location="../experiment_data/test_functions", numbers=[4])[0]
    np.save(str(MOCK_GT_FILE), f_im)


def make_mock_data(snr: float, theta_v: np.ndarray, y_file: Path, y_sd_file: Path):
    """
    Creates mock data that tries to be as similar to M54 data as possible.

    :param snr:
    :param ground_truth:
    """
    # Get ground truth.
    ground_truth = np.load(MOCK_GT_FILE)
    # Get M54 standard deviations.
    m54_data = uq4pk_src.data.M54()
    m54_data.logarithmically_resample(dv=50.)

    y_sd = m54_data.noise_level

    # Simulate data.
    fwdop = m54_setup_operator()
    y_bar = fwdop.fwd_unmasked(ground_truth.flatten(), theta_v)
    mask = fwdop.mask
    # Rescale y_sd in order to achieve roughly the desired signal-to-noise ratio FOR THE MASKED DATA.
    y_sd = y_sd * np.linalg.norm(y_bar[mask]) / (snr * np.linalg.norm(y_sd[mask]))
    # Simulate noise
    noise = np.random.randn(y_sd.size) * y_sd
    # Create mock data.
    y = y_bar + noise
    snr = np.linalg.norm(y[mask]) / np.linalg.norm(noise[mask])
    print(f"Exact SNR of simulated data: {snr}")

    # Test reconstruction.
    m54_model = m54_fit_model(y=y, y_sd=y_sd, theta_v=THETA_V)
    fitted_model = m54_model.fitted_model
    f_map = fitted_model.f_map.clip(min=0.)
    vmax = ground_truth.max()
    plot_distribution_function(image=ground_truth, show=True, vmax=vmax)
    plot_distribution_function(image=f_map, show=True, vmax=vmax)

    np.save(str(y_file), y)
    np.save(str(y_sd_file), y_sd)


#make_test_function()
make_mock_data(snr=1000, theta_v=THETA_V, y_file=MOCK1_FILE, y_sd_file=MOCK_SD1_FILE)
make_mock_data(snr=100, theta_v=THETA_V, y_file=MOCK2_FILE, y_sd_file=MOCK_SD2_FILE)



