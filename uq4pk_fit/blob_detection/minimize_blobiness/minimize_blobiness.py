
import numpy as np
from skimage import filters
from typing import Sequence, Union

from uq4pk_fit.uq_mode.linear_model import LinearModel
from ..detect_blobs import detect_blobs
from .scale_space_minimization import scale_space_minimization
from ..significant_blobs.detect_significant_blobs import _match_blobs


RTHRESH = 0.05
OTHRESH = 0.5


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
    f_min_pre = scale_space_minimization(alpha=alpha, m=m, n=n, model=model, x_map=x_map, sigma_list=sigma_list)
    f_min = filters.gaussian(f_min_pre, mode="constant", sigma=sigma_list[0])
    return f_min
