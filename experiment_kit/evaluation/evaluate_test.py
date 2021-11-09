
import numpy as np
import pandas
import skimage.metrics as scimet

from experiment_kit.test.test_result import TestResult
from ..data import EVALUATION_FILE, PARAMETER_FILE


def evaluate_testresult(savedir: str, test_result: TestResult):
    """
    Performs a numerical evaluation of a TestResult object, and stores it as "result.csv" in the given location.
    """
    # first, tabulate the parameters
    parameter_names = list(test_result.parameters.keys())
    parameter_values = list(test_result.parameters.values())
    quantity_names = ["rdm", "costmap", "costtruth", "errof", "ssimf", "errortheta", "errorvartheta"]
    # basic analysis
    error_quantities = list(_error_analysis(test_result))

    # analysis of uncertainty quantification
    uq_quantity_names = ["uqerrorf", "uqtightnessf", "uqsizef", "uqerrortheta", "uqtightnesstheta"]
    uq_error_quantities = list(_uq_error_analysis(test_result))

    names = parameter_names + quantity_names + uq_quantity_names
    # Use a Pandas dataframe for output to .csv-file.
    dataframe = pandas.DataFrame(columns=names)
    dataframe.loc[0] = parameter_values + error_quantities + uq_error_quantities
    # store the dataframe as csv-file
    dataframe.to_csv(f"{savedir}/{EVALUATION_FILE}")


def _error_analysis(tr: TestResult):
    """
    """
    # Compute relative data misfit
    f_map = tr.f_map
    f_ref = tr.data.f_ref
    f_true = tr.data.f_true
    theta_map = tr.theta_map
    theta_true = tr.data.theta_true
    rdm = tr.rdm_map
    rdm_ref = tr.rdm_ref
    rdm_truth = tr.rdm_true
    # Compare the cost functions
    m = 1e4
    cost_ref = tr.cost_ref / m
    cost_truth = tr.cost_true / m
    cost_map = tr.cost_map / m
    # Compute reconstruction error for f
    err_f, ssim_f = _compute_error_for_f(f_map, f_true, tr)
    err_ref, ssim_ref = _compute_error_for_f(f_ref, f_true, tr)
    # Compute reconstruction error for theta_v
    sre_theta, sre_theta_less = _compute_error_for_theta(theta_map, theta_true, regop_theta=tr.regop_theta)
    return rdm, cost_map, cost_truth, err_f, ssim_f, sre_theta, sre_theta_less


def _compute_error_for_f(f, f_true, tr: TestResult):
    """
    Performs complete error analysis for f, including uq if given.
    :return: float
        The relative reconstruction error of the normalized age-metallicity distribution.
    """
    err_f = _relative_error(map=f, truth=f_true)
    # Also compute SSIM
    f_true_im = tr.image(f_true)
    f_im = tr.image(f)
    try:
        ssim_f = scimet.structural_similarity(f_im, f_true_im,
                                              data_range=f_true_im.max() - f_true_im.min())
    except:
        ssim_f = -1
    return err_f, ssim_f


def _compute_error_for_theta(theta_map: np.ndarray, theta_true: np.ndarray, regop_theta: np.ndarray):
    """
    :return: The relative reconstruction error for theta.
    """
    # Compute errors
    sre_theta = _scaled_error(map=theta_map, truth=theta_true, s=regop_theta)
    less = [0, 1, 5, 6]
    idmat = np.eye(theta_map.size)
    emb_mat = idmat[:, less]
    regop_vartheta = regop_theta @ emb_mat
    sre_theta_less = _scaled_error(map=theta_map[less], truth=theta_true[less],
                                        s=regop_vartheta)
    return sre_theta, sre_theta_less


def _uq_error_analysis(testresult: TestResult):
    """
    :return: uqerr_f, uqtightness_f, uqerr_theta, uqtightness_theta
    """
    # Compute UQ error measures for f.
    uqerr_f, uqsize_f, uqtightness_f = _uq_error_analysis_f(testresult)
    uqerr_theta, uqtightness_theta = _uq_error_analysis_theta(testresult)
    return uqerr_f, uqtightness_f, uqsize_f, uqerr_theta, uqtightness_theta


def _uq_error_analysis_f(tr: TestResult):
    phi_true = tr.phi_true
    phi_map = tr.phi_map
    ci_f = tr.ci_f
    if ci_f is not None:
        uqerr_f = _uq_error(phi_true, ci_f)
        uqtightness_f = _uq_tightness(phi_map, phi_true, ci_f)
        ci_f_size = ci_f[:, 1] - ci_f[:, 0]
        uqsize_f = np.mean(ci_f_size.flatten())
    else:
        uqerr_f = -1
        uqtightness_f = -1
        uqsize_f = -1
    return uqerr_f, uqtightness_f, uqsize_f


def _uq_error_analysis_theta(tr: TestResult):
    ci_theta = tr.ci_theta
    if ci_theta is not None:
        theta_map = tr.theta_map
        theta_true = tr.data.theta_true
        r_theta = tr.regop_theta
        uqerr_theta = _uq_error(truth=theta_true, ci=ci_theta, scaling_matrix=r_theta)
        uqtightness_theta = _uq_tightness(x1=theta_true, x2=theta_map, ci=ci_theta, scaling_matrix=r_theta)
    else:
        uqerr_theta = -1
        uqtightness_theta = -1
    return uqerr_theta, uqtightness_theta


def _uq_error(truth: np.ndarray, ci: np.ndarray, scaling_matrix: np.ndarray = None):
    """
    Computes how much of x is actually inside the credible interval ci.

    Suppose a reference solution is given by x, and a credible interval is given by the vectors lb and ub.
    Then, the uncertainty error (scaled by a matrix S) is computed as:
    euq = ||(S(x - xi_low))^- + (S(x - xi_upp))^+||_2 / ||S x ||_2

    :param ci: The credible intervals as array of shape (n,2).
    :param scaling_matrix: The scaling matrix S. Defaults to the identity matrix if none is provided.
    :returns: The float ``euq``.
    """
    n = truth.size
    if scaling_matrix is None:
        s = np.identity(n)
    else:
        s = scaling_matrix
    too_large = s @ (truth - ci[:, 1])
    too_small = s @ (ci[:, 0] - truth)
    err_plus = _positive_part(too_large)
    err_minus = _positive_part(too_small)
    error = np.linalg.norm(np.concatenate((err_plus, err_minus))) / np.linalg.norm(s @ truth)
    return error


def _uq_tightness(x1: np.ndarray, x2: np.ndarray, ci: np.ndarray, scaling_matrix: np.ndarray = None):
    """
    Computes the tightness of the uncertainty quantification.
    Given two reference vectors x1 and x2 (e.g. x1 could be an estimate, and x2 would correspond to the ground truth),
    we compute how much the credible interval ``ci`` over-/underestimates the difference of x1 and x2.
    This is achieved by computing
    tuq = median(r),
    where r is the vector of ratios:
    r  = S (ub - lb) / (S(x1 - x2))
    """
    n = x1.size
    if scaling_matrix is None:
        s = np.identity(n)
    else:
        s = scaling_matrix
    diff = s @ (np.abs((x1 - x2)))
    ci_size = s @ (np.abs((ci[:, 1] - ci[:, 0])))
    # we compute the vector of ratios (clipping to avoid division by zero)
    ci_ratios = ci_size / diff.clip(min=1e-10)
    # then, we compute the median tightness
    tightness = np.median(ci_ratios)
    return tightness


def _relative_error(map, truth):
    return np.linalg.norm(map - truth) / np.linalg.norm(truth)


def _scaled_error(map, truth, s):
    return np.linalg.norm(s @ (map - truth)) / np.linalg.norm(s @ truth)


def _positive_part(x):
    """
    Returns the positive part of the vector x
    """
    return x.clip(min=0.)