
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from skimage import filters

from uq4pk_fit.uq_mode.linear_model import LinearModel
from ..detect_blobs import detect_blobs
from .scale_space_minimization import scale_space_minimization
from ..significant_blobs.detect_significant_blobs import _match_blobs


RTHRESH = 0.05
OTHRESH = 0.5


def minimize_blobiness(alpha: float, m: int, n: int, model: LinearModel, x_map: np.ndarray, sigma_min: float,
                       sigma_max: float, num_sigma: int = 10, ratio: float = 1.):
    """
    Detects "blobs of interest" by performing minimization in scale-space:

    .. math::

        \\min_f || \\nabla \\Delta_{norm} f ||_2^2      s. t. f \\in C_\\alpha.

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
        - f: Of shape (m, n). The "minimally blobby element".
    """
    # First, solve the scale-space optimization problem
    f_min_pre = scale_space_minimization(alpha=alpha, m=m, n=n, model=model, x_map=x_map, sigma_min=sigma_min,
                                     sigma_max=sigma_max, num_sigma=num_sigma, ratio=ratio)
    f_min = filters.gaussian(f_min_pre, mode="reflect", sigma=[ratio ** 2 * sigma_min, sigma_min])

    # Then, detect features in f_min
    boi = detect_blobs(image=f_min, sigma_min=sigma_min, sigma_max=sigma_max, num_sigma=num_sigma, max_overlap=OTHRESH,
                       rthresh=RTHRESH, mode="constant", ratio=ratio)
    # Also detect MAP features
    f_map = np.reshape(x_map, (m, n))
    map_blobs = detect_blobs(image=f_map, sigma_min=sigma_min, sigma_max=sigma_max, num_sigma=num_sigma,
                             max_overlap=OTHRESH, rthresh=RTHRESH, mode="constant", ratio=ratio)

    # Perform feature matching.
    blob_pairs = _match_blobs(significant_blobs=boi, map_blobs=map_blobs, overlap=)

    return blob_pairs
