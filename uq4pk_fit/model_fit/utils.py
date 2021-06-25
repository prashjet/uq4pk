"""
Contains functions "get_f", "plot_doublebar", "plot_with_colorbar", "error_analysis", "basic_plotting"
and "sample_theta".
"""

import numpy as np

from uq4pk_fit.regop import DiagonalOperator


def get_f(location, maxno=None):
    """
    Reads the stored test functions from the csv files and returns them as list of numpy arrays.
    :return: list
    """
    f1 = np.loadtxt(f'{location}/f0.csv', delimiter=',')
    f2 = np.loadtxt(f'{location}/f1.csv', delimiter=',')
    f3 = np.loadtxt(f'{location}/f2.csv', delimiter=',')
    flist = [f1, f2, f3]
    if maxno is None:
        return_list = flist
    else:
        return_list = flist[:maxno]
    return return_list


def error_analysis(y, y_noi, f_true, f_map, theta_v_true, costfun, misfit, theta_v_map=None):
    """
    :param y: numpy vector
    :param y_noi: numpy vector
    :param f_true: numpy vector
    :param f_map: numpy vector
    :param theta_v_true: numpy vector
    :param costfun: function
        The MAP cost function. Takes either two or one argument, depending on the model.
    :param misfit: function
        The misfit function y-fwd(.). Takes either two or one argument, depending on the model.
    :param theta_v_map: numpy vector, optional
        If provided, assume nonlinear model.
        If not provided, assume linear model (theta_v fixed)
    """
    # start error analysis
    print(" ")
    print("ERROR ANALYSIS")
    if theta_v_map is None:
        theta_fixed = True
    else:
        theta_fixed = False
    # compute relative data misfit
    if theta_fixed:
        map_misfit = misfit(f_map)
    else:
        map_misfit = np.linalg.norm(misfit(f_map, theta_v_map))
    noise_level = np.linalg.norm(y_noi - y)
    print(f"||y_noi - y_map|| / ||y_noi - y||: {map_misfit / noise_level}")
    # Compare the cost functions
    if theta_fixed:
        cost_truth = costfun(f_true)
        cost_map = costfun(f_map)
    else:
        cost_truth = costfun(f_true, theta_v_true)
        cost_map = costfun(f_map, theta_v_map)
    print(f"Cost at true parameter: {cost_truth}")
    print(f"Cost at MAP estimate: {cost_map}")
    # Error analysis for f:
    print("Error analysis for f:")
    # Let us print the relative reconstruction errors
    relerr_f = np.linalg.norm(f_map - f_true) / np.linalg.norm(f_true)
    print(f"Relative reconstruction error of f: {relerr_f}")
    # Let us also compare the maximum values
    print(f"max(f)={np.max(f_true)}")
    print(f"max(f_map)={np.max(f_map)}")
    print(" ")
    # Let us also print the maximum deviation
    maxdev_f = np.max(np.abs(f_map - f_true))
    print(f"Maximum deviation between f_map and f_true: {maxdev_f}")
    # if theta_v is not fixed, perform also error analysis for theta_v:
    if not theta_fixed:
        # Error analysis for theta_v:
        print("Error analysis for theta_v:")
        relerr_theta_v = np.linalg.norm(theta_v_map - theta_v_true) / np.linalg.norm(theta_v_true)
        print(f"Relative reconstruction error of theta_v: {relerr_theta_v}")
        maxdev_theta_v = np.max(np.abs(theta_v_map - theta_v_true))
        print(f"Maximum deviation of theta_v: {maxdev_theta_v}")


def sample_theta(q, theta_v):
    """
    Generates a noisy guess of theta_v
    :param q: noise percentage
    :param theta_v: the true value of theta_v
    :return: ndarray (n,), ndarray (n,n)
        The initial guess and the corresponding regularization operator.
    """
    n_theta = theta_v.size
    horder = n_theta - 2
    v_sigma_stdev = np.array([30., 100.])
    h_stdev = np.ones(horder)
    v_stdev = np.concatenate((v_sigma_stdev, h_stdev))
    noise_stdev = q * v_stdev
    noise = noise_stdev * np.random.randn(n_theta)
    theta_v_guess = theta_v + noise
    # we also set the prior covariance equal to the covariance of (theta_v_guess - theta_v)
    regop = DiagonalOperator(dim=n_theta, s=np.divide(1, noise_stdev))
    return theta_v_guess, regop
