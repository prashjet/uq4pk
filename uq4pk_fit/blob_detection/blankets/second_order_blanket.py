
import numpy as np
from typing import Literal

from uq4pk_fit.special_operators import DiscreteLaplacian
from .minimize_surface import minimize_surface


def second_order_blanket(lb: np.ndarray, ub: np.ndarray, mode: Literal["fast", "exact"] = "fast"):
    """
    Computes the second-order blanket between two d-dimensional signals. That is, the optimization problem

    .. math::
        min_f \\qquad \\sum_{ij} \\sqrt{1 + ||\\nabla G f_{ij}||_2^2}

        s. t. \\qquad l \\leq f \\leq u.

    is solved.

    :param lb: The lower bound.
    :param ub: The upper bound.
    :param mode: The mode for solution. If "fast", then the simpler optimization problem

        .. math::
            min_f \\qquad \\sum_{ij} ||\\nabla \\delta f_{ij}||_2^2
            s. t. \\qquad l \\leq f \\leq u.

        is solved.

    :return: The solution, as d-dimensional array of the same shape as 'lb' and 'ub'.
    """
    assert lb.shape == ub.shape

    # Initialize the discrete Laplacian.
    delta = DiscreteLaplacian(shape=lb.shape, mode="reflect").mat

    # Compute second-order blanket with the method of choice.
    blanket = minimize_surface(lb=lb, ub=ub, g=delta, mode=mode)

    return blanket
