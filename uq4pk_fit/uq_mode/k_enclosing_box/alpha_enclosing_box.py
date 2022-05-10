
import numpy as np

from .naive_k_enclosing_box import naive_k_enclosing_box


def alpha_enclosing_box(alpha: float, points: np.ndarray) -> np.ndarray:
    """
    Given n d-dimensional points, finds smallest box that contains ceil((1 - alpha) * n) points.

    :param alpha: A number between 0 and 1.
    :param points: Of shape (n, d), where n is the number of points and d their dimension.
    :return: box. An array of shape (2, d), that determines a box in R^d via the condition box[0] <= x <= box[1].
    """
    assert 0 <= alpha <= 1
    assert points.ndim == 2

    n = points.shape[0]
    k = np.ceil((1 - alpha) * n).astype(int)
    alpha_box = naive_k_enclosing_box(k, points)
    return alpha_box


