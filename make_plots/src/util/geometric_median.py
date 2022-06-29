"""
file: geometric_median.py
"""


import numpy as np
from scipy.spatial.distance import cdist, euclidean


MAXITER = 5000      # Maximum number of iterations until we give up.


def geometric_median(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Computes the geometric median (see https://en.wikipedia.org/wiki/Geometric_median).
    Code copied from StackOverflow:
        https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points


    :param X: Of shape (n, ...), corresponding to n points of dimension d.
    :param eps: Minimum accuracy for the solution.
    :returns y: Of shape (...). The geometric median.
    """
    assert X.ndim >= 2
    if X.ndim > 2:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    # Remove scale.
    scale = np.max(X)
    X_flat /= scale

    y = np.mean(X_flat, 0)
    success = False
    for i in range(MAXITER):
        D = cdist(X_flat, [y])
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X_flat[nonzeros], 0)
        num_zeros = len(X_flat) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X_flat):
            y_out = y
            success = True
            break
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y
        if euclidean(y, y1) < eps:
            y_out = y1
            success = True
            break
        y = y1
    if not success:
        print("WARNING: Geometric median did not converge.")
        y_out = y1

    # Bring back to original scale.
    y_out *= scale

    assert y_out is not None
    # Postprocess in case X has more than two dimensions.
    return y_out.reshape(X[0].shape)