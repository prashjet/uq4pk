
import numpy as np
from skimage import filters
from typing import Sequence, Union

from uq4pk_fit.uq_mode.linear_model import LinearModel
from .minimal_representation import minimal_representation
from .scale_space_minimization import scale_space_minimization

RTHRESH = 0.05
OTHRESH = 0.5

old_mode = False


def minimize_blobiness(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray,
                       sigma_list: Sequence[Union[float, np.ndarray]]):
    """
    Detects "blobs of interest" by performing minimization in scale-space:

    .. math::

        \\min_f || \\nabla \\Delta_{norm} f ||_2^2      s. t. f \\idim C_\\alpha.

    :param alpha:
    :param m:
    :param n:
    :param model:
    :param x_map:
    :param sigma_list: List of standard deviations for Gaussian kernels.
    :return: f_min
    """
    # First, solve the scale-space optimization problem
    if old_mode:
        f_min_pre = scale_space_minimization(alpha=alpha, m=m, n=n, model=model, x_map=x_map, sigma_list=sigma_list)
        f_min = filters.gaussian(f_min_pre, mode="reflect", sigma=sigma_list[0])
    else:
        # For every scale, minimize the Laplacian.
        l_list = []
        for sigma in sigma_list:
            l_t = minimal_representation(alpha=alpha, m=m, n=n, model=model, x_map=x_map, sigma=sigma)
            l_list.append(l_t)
        f_min = np.array(l_list)


    return f_min
