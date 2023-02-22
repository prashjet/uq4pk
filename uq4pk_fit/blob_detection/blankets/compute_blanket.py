import numpy as np

from uq4pk_fit.optimization import cls_solve, CLS
from uq4pk_fit.operators import DiscreteLaplacian


def compute_blanket(lb: np.ndarray, ub: np.ndarray):
    """
    Solves minimize_B ||scaled_laplacian(B)||_2^2 s. t. lb <= B <= ub.
    Parameters
    ----------
    lb : shape (m, n)
    ub : shape (m, n)

    Returns
    -------
    blanket : shape (m, n)
        The solution to the minimization problem.
    """
    assert lb.shape == ub.shape
    assert np.all(lb <= ub)
    # If ub > lb, then we can safely return the zero blanket.
    if ub.min() > lb.max():
        blanket = np.ones(lb.shape) * lb.max()
        return blanket

    # Initialize the discrete Laplacian.
    delta = DiscreteLaplacian(shape=lb.shape).mat

    # First, rescale.
    scale = ub.max()

    lbvec = lb.flatten() / scale
    ubvec = ub.flatten() / scale
    n = lbvec.size

    # Define CLS.
    cls = CLS(h=delta, y=np.zeros(n), lb=lbvec, ub=ubvec)

    # Solve with `cls_solve`
    x_min = cls_solve(cls)

    # Bring minimizer back to the original scale.
    x_min = scale * x_min

    # Bring minimizer into the correct format.
    blanket = np.reshape(x_min, lb.shape)

    # Return the solution as two-dimensional numpy array.
    return blanket
