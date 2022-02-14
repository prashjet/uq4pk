
import numpy as np

from uq4pk_fit.visualization import plot_theta


def test_plot_theta():
    # Create fake results.
    names = ["h_0", "h_1", "h_2", "h_3"]
    theta_true = np.array([1., 0, -0.5, 0.1])
    theta_map = theta_true + 0.05 * np.random.randn(4)
    theta_guess = theta_true + np.random.randn(4)
    l_theta = theta_map - 0.1
    u_theta = theta_map + 0.1
    ci_theta = np.column_stack([l_theta, u_theta])

    # Plot them.
    plot_theta(theta_map=theta_map, names=names, theta_guess=theta_guess, theta_true=theta_true, ci_theta=ci_theta)
