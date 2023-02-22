
import numpy as np

from .naive_k_enclosing_box import naive_k_enclosing_box


def alpha_enclosing_box(alpha: float, points: np.ndarray) -> np.ndarray:
    """
    Given n d-dimensional points, finds smallest box that contains ceil((1 - alpha) * n) points
    """
    assert 0 <= alpha <= 1
    assert points.ndim == 2

    n = points.shape[0]
    k = np.ceil((1 - alpha) * n).astype(int)
    alpha_box = naive_k_enclosing_box(k, points)
    return alpha_box


