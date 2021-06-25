"""
Contains plotting functions
"""

from matplotlib import pyplot as plt
import numpy as np


def plot_everything(savename, f_true_im, f_map_im, theta_v_true, theta_v_map=None,  uq=None):
    """
    :param savename: string
        Name of directory in which results will be stored
    :param f_true_im: ndarray
        True age-metallicity distribution
    :param f_map_im: ndarray
        MAP estimate of f
    :param theta_v_true: ndarray
        True value of the parameters for the Gauss-Hermite expansion
    :param theta_v_map: ndarray, optional
        MAP estimate of theta_v. If not provided, we assume that theta_v is fixed and evaluate
        the model accordingly.
    :param uq: dict, optional
        If provided, the function will also plot the uncertainty quantification for everything.
        uq must have the following keys:
        - "lci_f":  The local credible intervals for the distribution function f.
        - "lci_theta_v": The local credible itnervals for theta_v. Only has to be provided if theta_v_map is given.
    """
    # if theta_v_map is not provided, we treat theta_v as fixed to theta_v_true
    dim_f = f_true_im.size
    if theta_v_map is None:
        theta_fixed = True
    else:
        theta_fixed = False
    _plot_f(savename=savename, f_true=f_true_im, f_map=f_map_im, uq=uq)
    if not theta_fixed:
        _plot_theta_v(savename, theta_v_true, theta_v_map, dim_f, uq)


def _plot_f(savename, f_true, f_map, uq):
    fall = np.concatenate((f_true.flatten(), f_map.flatten()))
    vmax = np.max(fall, axis=0)
    # Plot true distribution function vs MAP estimate
    plot_with_colorbar(image=f_true, savename=f"{savename}/truth.png", vmax=vmax)
    plot_with_colorbar(image=f_map, savename=f"{savename}/map.png", vmax=vmax)
    dim_f1 = f_true.shape[0]
    dim_f2 = f_true.shape[1]
    dim_f = f_true.size
    if uq is not None:
        lci = uq["lci_f"]
        # visualize uncertainty quantification for f
        fmin = np.reshape(lci[:dim_f, 0], (dim_f1, dim_f2))
        fmax = np.reshape(lci[:dim_f, 1], (dim_f1, dim_f2))
        savename1 = f"{savename}/f_lower.png"
        savename2 = f"{savename}/f_upper.png"
        # determine vmax
        vmax = np.max(lci.flatten())
        # plot the lower bound
        plot_with_colorbar(image=fmin, savename=savename1, vmax=vmax)
        plot_with_colorbar(image=fmax, savename=savename2, vmax=vmax)


def _plot_theta_v(savename, theta_v_true, theta_v_map, dim_f, uq):
    # Plot true theta_v vs MAP theta_v
    # separate plot for first_two_values
    if uq is not None:
        # if we want uncertainty quantification, we compute the error bars from the local credible intervals
        lci_theta_v = uq["lci_theta_v"]
        theta_v_min = lci_theta_v[:, 1]
        theta_v_max = lci_theta_v[:, 0]
        # errorbars are the differences between x1 and x2
        below_error = theta_v_map - theta_v_min
        upper_error = theta_v_max - theta_v_map
        errorbars = np.row_stack((below_error, upper_error))
        errorbars1 = errorbars[:, :2]
        errorbars2 = errorbars[:, 2:]
    else:
        errorbars1 = None
        errorbars2 = None
    savename1 = f"{savename}/V_and_sigma.png"
    savename2 = f"{savename}/h.png"
    names1 = ["V", "sigma"]
    plot_double_bar(safename=savename1, name_list=names1, values1=theta_v_true[0:2],
                    values2=theta_v_map[0:2], name1="Truth", name2="MAP estimate", errorbars=errorbars1)
    # plot remaining variables
    # assemble list of names h_0, ... h_m, depending on m
    m = theta_v_map.size - 2
    h_names = []
    for i in range(m):
        h_names.append(f"h_{i}")
    plot_double_bar(safename=savename2, name_list=h_names, values1=theta_v_map[2:],
                    values2=theta_v_map[2:], name1="Truth", name2="MAP estimate", errorbars=errorbars2)


def plot_double_bar(safename, name_list, values1, values2, name1, name2, errorbars=None):
    """
    Creates a double-bar chart with error bars.
    :param safename: string
    :param name_list: list
    :param values1: ndarray
    :param values2: ndarray
    :param name1: string
    :param name2: string
    :param errorbars: ndarray
    """
    bar_width = 0.4
    grid = np.arange(len(name_list))
    fig = plt.figure()
    plt.bar(grid - 0.5 * bar_width, values1, bar_width, label=name1)
    if errorbars is None:
        plt.bar(grid + 0.5 * bar_width, values2, bar_width, label=name2)
    else:
        plt.bar(grid + 0.5 * bar_width, values2, bar_width, label=name2, yerr=errorbars, capsize=5)
    plt.xticks(grid, name_list)
    plt.xlabel("Parameter")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(safename, bbox_inches="tight")
    plt.close()


def plot_with_colorbar(image, savename, vmax=None):
    """
    Plots the age-metallicity distribution function with a colorbar on the side that
    shows which color belongs to which value.
    :param image: ndarray
    :param savename: string
    :param vmax: float
    """
    cmap = plt.get_cmap("gnuplot")
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(image, vmax=vmax, cmap=cmap)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.xlabel("Age")
    plt.ylabel("Metallicity")
    plt.savefig(savename, bbox_inches='tight')
    plt.close()
