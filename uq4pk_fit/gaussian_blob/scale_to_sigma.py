
import numpy as np


def sigma_to_scale2d(sigma: np.ndarray) -> float:
    t = np.square(sigma[1])
    return t
