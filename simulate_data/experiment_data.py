
import numpy as np

from uq4pk_fit.inference import ForwardOperator


class ExperimentData:
    """
    Assembles all relevant parameters for a simulated experiment.
    :attr snr: The signal-to-noise ratio.
    :attr y: the exact measurement
    :attr y_noi: the noisy measurement
    :attr f: the true distribution function from which y is generated
    :attr theta_v: the true value of theta_v from which y is generated
    :attr sdev: the standard deviation of the noise in y_noi
    :attr ssps: the sampling grid for the distribution function (see ObservationOperator.ssps)
    """
    def __init__(self, name: str, snr: float, y: np.ndarray, y_sd: np.ndarray, f_true: np.ndarray, f_ref: np.ndarray,
                 theta_true: np.ndarray, theta_guess: np.ndarray, theta_sd: np.ndarray,
                 forward_operator: ForwardOperator, theta_noise: np.ndarray):
        self.name = name
        self.snr = snr
        self.setup = {"theta_noise": theta_noise}
        self._check_input(snr, y, y_sd, f_true, theta_true, theta_guess, theta_sd, forward_operator)
        self.y = y
        self.f_true = f_true
        self.f_ref = f_ref
        self.theta_true = theta_true
        self.theta_guess = theta_guess
        self.theta_sd = theta_sd
        self.theta_noise = theta_noise
        self.y_sd = y_sd
        self.forward_operator = forward_operator
        self.ssps = self.forward_operator.modgrid

    @staticmethod
    def _check_input(snr, y, y_sd, f_true, theta_true, theta_guess, theta_sd, forward_operator):
        if snr <= 0:
            raise ValueError("Non-positive SNR makes no sense!")
        theta_good = (theta_true.size == theta_guess.size == theta_sd.size)
        # check that the forward operator matches the dimensions of f and y
        dimensions_match = (forward_operator.dim_f == f_true.size and forward_operator.dim_y == y.size
                            and forward_operator.dim_theta == theta_true.size and y.size == y_sd.size)
        # check that no of the provided parameters contain NaNs or infs.
        some_is_nan = False
        some_is_inf = False
        for arr in [y, f_true, theta_true, theta_guess, theta_sd]:
            if np.isnan(arr).any():
                some_is_nan = True
            if np.isinf(arr).any():
                some_is_inf = True
        # also, y_sd must not be zero or negative
        y_sd_good = np.all(y_sd > 1e-16)
        print(f"y_sd_min = {np.min(y_sd)}")
        if not (theta_good and dimensions_match):
            raise ValueError("Inconsistent dimensions.")
        if some_is_nan:
            raise ValueError("NaN-values not allowed in data.")
        if some_is_inf:
            raise ValueError("Inf-values not allowed in data.")
        if not y_sd_good:
            raise ValueError("y_sd must be strictly positive.")
