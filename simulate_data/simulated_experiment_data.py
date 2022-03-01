
import numpy as np
import os
from pathlib import Path

from uq4pk_fit.inference import ForwardOperator
from uq4pk_src import model_grids
from .experiment_data import ExperimentData


class SimulatedExperimentData(ExperimentData):
    """
    Assembles all relevant parameters for a simulated experiment.
    :ivar name: A string used as identifier.
    :ivar snr: The signal-to-noise ratio.
    :ivar y: The noisy measurement, without the masked values!
    :ivar y_sd: The vector of standard deviations of the measurement noise, without the masked values!
    :ivar f_true: The ground truth for the distribution function.
    :ivar f_ref: A reference distribution function, for example the ppxf-estimate.
    :ivar theta_true: The ground truth for the parameter theta_v.
    :ivar theta_guess: A corresponding initial guess for theta_v.
    :ivar theta_sd: The prior standard deviations for theta_v.
    :ivar hermite_order: The order of the Gauss-Hermite expansion.
    :ivar mask: A vector of the same length as `y`. If mask[i]=1, then y[i] is included in the inference. If mask[i]=0,
        it is ignored.
    :ivar grid_type: A string denoting the used grid.
    """

    _possible_grid_types = ["MILES", "EMILES"]

    def __init__(self, name: str, snr: float, y: np.ndarray, y_sd: np.ndarray, f_true: np.ndarray, f_ref: np.ndarray,
                 theta_true: np.ndarray, theta_guess: np.ndarray, theta_sd: np.ndarray, hermite_order: int,
                 mask: np.ndarray = None, grid_type: str = "MILES"):
        # CHECK INPUT FOR CONSISTENCY
        assert isinstance(name, str)
        if snr <= 0:
            raise ValueError("Non-positive SNR makes no sense!")
        assert y.ndim == 1
        assert y.shape == y_sd.shape
        assert theta_true.size == theta_guess.size == theta_sd.size
        assert theta_true.size == hermite_order + 3
        # check that no of the provided parameters contain NaNs or infs.
        some_is_nan = False
        some_is_inf = False
        for arr in [y, f_true, theta_true, theta_guess, theta_sd]:
            if np.isnan(arr).any():
                some_is_nan = True
            if np.isinf(arr).any():
                some_is_inf = True
        assert not some_is_nan
        assert not some_is_inf
        # also, y_sd must not be zero or negative
        assert np.all(y_sd > 1e-16)
        if mask is None:
            self.mask = np.full((y.size,), True, dtype=bool)
        else:
            # mask must have same shape as y
            assert mask.size >= y.size
            assert mask.ndim == 1
            self.mask = mask
        assert grid_type in self._possible_grid_types

        # Set instance variables.
        self.name = name
        self.snr = snr
        self.y = y
        self.f_true = f_true
        self.f_ref = f_ref
        self.theta_ref = theta_true
        self.theta_guess = theta_guess
        self.theta_sd = theta_sd
        self.y_sd = y_sd
        self.hermite_order = hermite_order
        self.grid_type = grid_type

    @property
    def ssps(self):
        """
        Returns the model grid.
        """
        ssps_pre = self._ssps_pre
        # Let forward operator initialize ssps (HACK)
        fwdop = ForwardOperator(hermite_order=self.hermite_order, mask=self.mask, ssps=ssps_pre)
        ssps = fwdop.modgrid
        return ssps

    @property
    def forward_operator(self) -> ForwardOperator:
        """
        Creates the model forward operator.
        """
        ssps_pre = self._ssps_pre
        op = ForwardOperator(hermite_order=self.hermite_order, mask=self.mask, ssps=ssps_pre, do_log_resample=False,
                             dv=ssps_pre.dv)
        return op

    @property
    def _ssps_pre(self):
        if self.grid_type == self._possible_grid_types[0]:
            ssps_pre = model_grids.MilesSSP()
        elif self.grid_type == self._possible_grid_types[1]:
            ssps_pre = model_grids.MilesSSP(
                miles_mod_directory='EMILES_BASTI_BASE_BI_FITS',
                imf_string='Ebi1.30',
                lmd_min=None,
                lmd_max=None,
            )
        else:
            raise NotImplementedError("This should not happen!")
        return ssps_pre


def save_experiment_data(data: SimulatedExperimentData, savename: str):
    """
    Stores experiment data as a pickle-file.

    :param data: The experiment data.
    :param savename: The relative path to the desired save location.
    """
    # Create folder if it does not exist already.
    Path(savename).mkdir(parents=True, exist_ok=True)

    # Store all variables as .npy files.
    def quicksave(arr, name):
        np.save(arr=arr, file=os.path.join(savename, name))

    info_array = np.array([data.name, data.grid_type])
    quicksave(info_array, "info.npy")
    quicksave(np.array(data.snr), "snr.npy")
    quicksave(np.array(data.hermite_order), "hermite_order.npy")
    quicksave(data.y, "y.npy")
    quicksave(data.y_sd, "y_sd.npy")
    quicksave(data.f_true, "f_true.npy")
    quicksave(data.f_ref, "f_ref.npy")
    quicksave(data.theta_ref, "theta_ref.npy")
    quicksave(data.theta_sd, "theta_sd.npy")
    quicksave(data.theta_guess, "theta_guess.npy")
    quicksave(data.mask, "mask.npy")


def load_experiment_data(savedir: str) -> SimulatedExperimentData:
    """
    Loads an :py:class:`ExperimentData` object from a stored file.

    :param savedir: Name of the folder where the data is stored.
    :return: The ExperimentData object.
    """

    def quickload(fname: str):
        return np.load(os.path.join(savedir, fname), allow_pickle=True)

    # Load individual components.
    info_array = quickload("info.npy")
    snr = quickload("snr.npy")
    hermite_order = quickload("hermite_order.npy")
    y = quickload("y.npy")
    y_sd = quickload("y_sd.npy")
    f_true = quickload("f_true.npy")
    f_ref = quickload("f_ref.npy")
    theta_true = quickload("theta_true.npy")
    theta_sd = quickload("theta_sd.npy")
    theta_guess = quickload("theta_guess.npy")
    mask = quickload("mask.npy")

    # From the loaded components, create the corresponding ExperimentData object.
    data = SimulatedExperimentData(name=info_array[0], snr=snr, hermite_order=hermite_order, grid_type=info_array[1],
                                   y=y, y_sd=y_sd, f_true=f_true, f_ref=f_ref, theta_true=theta_true, theta_sd=theta_sd,
                                   theta_guess=theta_guess, mask=mask)

    return data
