
from copy import deepcopy
import numpy as np
from skimage.feature import peak_local_max
from typing import List, Union

import uq4pk_fit.uq_mode as uq_mode
from .make_filter_function import make_filter_function
from .uq_result import UQResult


def autodetect(m: int, n: int, x_map: np.ndarray, model: uq_mode.LinearModel, options: dict) -> UQResult:
    """
    Automatically detects components of the age-metallicity distribution.
    :param options: A dictionary with further options:
        - "kernel": The filter kernel. Possible values are "laplace" and "gauss". Default is "laplace".
        - "minscale": The minimum resolution that is tried. Default is 1.
        - "maxscale": The maximum resolution that is tried. Default is 5.
    :return: lower, upper, scale, filter
        ``lower`` and ``upper`` are the FCI lower and upper bounds at the last scale, ``blobs`` is an (2, k)-array
        where each column corresponds to the coordinates of the found components. ``scale`` is the resolution at
        which the components where identified.
    """
    # Check matching
    assert model.n == m * n
    # Read options
    kernel = options.setdefault("kernel", "laplace")
    # Set starting scale
    h_min = options.setdefault("minscale", 3.)
    h_max = options.setdefault("maxscale", 10.)
    h_tol = options.setdefault("scaletol", 1.)
    maxiter = options.setdefault("maxiter", 10)
    # For each scale, compute the credible intervals
    h = 0.5 * (h_max + h_min)
    counter = 0
    agreement_found = True
    while h_max - h_min > h_tol:
        if counter >= maxiter:
            agreement_found = False
            break
        print(f"SCALE = {h}")
        fci_options = {"kernel": kernel, "h": h}
        # Create appropriate filter
        filter_function, filter_f, filter_vartheta, filter_theta = make_filter_function(m_f=m,
                                                                                        n_f=n,
                                                                                        options=fci_options)
        # compute filtered credible intervals
        ci = uq_mode.fci(alpha=0.05, x_map=x_map, model=model,
                           ffunction=filter_function,
                           options=options)
        lower = ci[:, 0]
        upper = ci[:, 1]
        # Reshape the lower and upper bound to images.
        lower_im = np.reshape(lower, (m, n))
        upper_im = np.reshape(upper, (m, n))
        # Detect the components according to the lower bound.
        # Orient on upper_im scale
        treshold = upper.max() * 0.01
        components_low = _detect_components(lower_im, h, treshold)
        # Detect the components according to the upper bound.
        components_upper = _detect_components(upper_im, h, treshold)
        # Check whether the components agree. If yes, break.
        if _components_agree(components_low, components_upper, h):
            # in that case, the optimal h must be between h_low and h
            h_max = h
            h = 0.5 * (h + h_min)
        else:
            # in this case, the optimal h must be between h and h_max
            h_min = h
            h = 0.5 * (h_max + h)
    if not agreement_found:
        raise Warning("NO AGREEMENT FOUND")
    # Make the corresponding UQResult object
    return lower, upper, h, filter_f


def _detect_components(image: np.ndarray, scale: float, treshold: float) -> Union[np.ndarray, None]:
    """
    Detects components of an image by identifying the local maxima.

    :param image: The image.
    :param scale: The scale at which the components are to be detected.
    :param treshold: The treshold for the intensity of the components.
    :return: An array of shape (2, k), where each column denotes the coordinates of the identified component.
        Returns None if no components are found.
    """
    m, n = image.shape
    int_scale = np.ceil(scale).astype(int)
    patched_image = np.zeros((m + 2 * int_scale, n + 2 * int_scale))
    patched_image[int_scale:-int_scale, int_scale:-int_scale] = image
    # Detect local maxima
    coordinates = peak_local_max(patched_image, min_distance=int_scale, threshold_abs=treshold)
    # Translate the coordinates to coordinates of the original image
    coordinates = coordinates - int_scale
    return coordinates.T


def _components_agree(components1: np.ndarray, components2: np.ndarray, scale: float) -> bool:
    """
    Checks whether two sets of components agree. Agreeing means the following: For every component of the first set
    there is only one component in the second set that is closer to it than the given resolution.

    :param components1: Array of shape (2, k1), where every column corresponds to a component. The components must
        not be ordered.
    :param components2: Array of shape (2, k2). Note that if k1 != k2, then the function immediately returns
        ``False``.
    :param scale: The resolution.
    :return: True, if all components agree. False, else.
    """
    # The given arrays must have the right format.
    assert components1.shape[0] == 2 == components2.shape[0]
    # The number of components must agree.
    if components1.shape[1] != components2.shape[1]:
        return False
    else:
        # Go through the first component array and try to find for each component a matching partner.
        k = components1.shape[1]
        for i in range(k):
            comp_i = components1[:, 0]
            j_i = _find_match(comp_i, components2, scale)
            # If no match is found, return False
            if j_i is None:
                return False
            else:
                # Delete corresponding columns from components1 and components2
                components1 = np.delete(components1, 0, axis=1)
                components2 = np.delete(components2, j_i, axis=1)
        return True


def _find_match(vec: np.ndarray, arr: np.ndarray, res: float) -> Union[int, None]:
    """
    Finds a matching vector in a given array.
    :param vec:
    :param arr:
    :return: If found, the index of the matching vector. Otherwise, returns None.
    """
    j_match = None
    for j in range(arr.shape[1]):
        vec_j = arr[:, j]
        if _vector_equal(vec, vec_j, res):
            j_match = j
            break
    return j_match


def _vector_equal(vec1: np.ndarray, vec2: np.ndarray, res: float):
    """
    Two vectors are equal if their l2-distance is less or equal to resolution.
    :param vec:
    :param vec_j:
    :param res:
    :return: True if equal, else False.
    """
    dist = np.linalg.norm(vec1 - vec2)
    close_enough = (dist <= res)
    return close_enough