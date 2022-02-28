
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
    :param sigma_min:
    :param sigma_max:
    :param num_sigma:
    :param ratio:
    :return: f, boi
        - f: Of shape (m, dim). The "minimally blobby element".
    """
    # First, solve the scale-space optimization problem
    f_min_pre = scale_space_minimization(alpha=alpha, m=m, n=n, model=model, x_map=x_map, sigma_list=sigma_list)
    f_min = filters.gaussian(f_min_pre, mode="reflect", sigma=sigma_list[0])

    # Then, detect features in f_min
    boi = detect_blobs(image=f_min, sigma_list=sigma_list, max_overlap=OTHRESH, rthresh=RTHRESH, mode="constant")
    # Also detect MAP features
    f_map = np.reshape(x_map, (m, n))
    map_blobs = detect_blobs(image=f_map, sigma_list=sigma_list, max_overlap=OTHRESH, rthresh=RTHRESH, mode="constant")

    # Perform feature matching.
    blob_pairs = _match_blobs(significant_blobs=boi, map_blobs=map_blobs, overlap=OTHRESH)

    return blob_pairs
