import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence


def plot_theta(theta_map: np.ndarray, ci_theta: np.ndarray = None, names: Sequence[str] = None,
               theta_guess: np.ndarray = None, theta_true: np.ndarray = None, savefile: str = None,
               show: bool = True):
    """
    Makes a single-, double-, or triple-bar plot of theta, optionally together with uncertainty quantification and
    values of the ground truth and initial guess.

    :param theta_map: The MAP estimate of theta.
    :param ci_theta: Optional. If provided, must be an array of shape (d, 2), where d=theta_map.size, and the columns
        correspond to the lower resp. upper bound of the credible intervals.
    :param names: Optional. Contains names for the individual components of theta. Must have same length as theta_map.
    :param theta_guess: Optional. The initial guess for theta. Must have the same shape as theta_map.
    :param theta_true: Optional. The true value of theta. Must have the same shape as theta_map.
    :param savefile: Optional. If provided, the plot is saved under this location.
    :param show: If set to False, nothing is plotted. Defaults to True.
    :return:
    """
    # Check that the input is consistent.
    if theta_map.ndim != 1:
        raise ValueError("'theta_ma' must be a numpy vector.")
    if ci_theta is not None:
        if ci_theta.shape != (theta_map.size, 2):
            raise ValueError(f"'theta_ci' must have shape ({theta_map.size}, 2).")
    if names is not None:
        if len(names) != theta_map.size:
            raise ValueError(f"'names' must have length {theta_map.size}.")
    if theta_guess is not None:
        if theta_guess.shape != theta_map.shape:
            raise ValueError("'theta_guess' must have same shape as 'theta_map'.")
    if theta_true is not None:
        if theta_true.shape != theta_map.shape:
            raise ValueError("'theta_true' must have same shape as 'theta_map'.")

    # Make the plot
    bar_width = 0.2
    grid = np.arange(len(names))
    fig = plt.figure()
    # Plot reference values
    if theta_guess is not None:
        plt.bar(grid - bar_width, theta_guess, bar_width, label="Initial guess")
    # Plot MAP (optionally with errorbars)
    if ci_theta is None:
        plt.bar(grid, theta_map, bar_width, label=names)
    else:
        below_error = theta_map - ci_theta[:, 0]
        upper_error = ci_theta[:, 1] - theta_map
        errorbars = np.row_stack((below_error, upper_error))
        plt.bar(grid, theta_map, bar_width, label="MAP estimate", yerr=errorbars, capsize=5)
    # Plot ground truth
    if theta_true is not None:
        plt.bar(grid + bar_width, theta_true, bar_width, label="Ground truth")
    plt.xticks(grid, names)
    plt.xlabel("Parameter")
    plt.ylabel("Value")
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()