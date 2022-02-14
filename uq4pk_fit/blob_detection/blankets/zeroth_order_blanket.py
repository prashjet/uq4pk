import numpy as np
from typing import Literal

from .minimize_surface import minimize_surface


def zeroth_order_blanket(lb: np.ndarray, ub: np.ndarray, mode: Literal["fast", "exact"] = "fast"):
    """
    Computes the zeroth-order blanket between two d-dimensional signals. That is, the optimization problem

    .. math::
        min_f \\qquad \\sum_{ij} \\sqrt{1 + ||\\nabla f_{ij}||_2^2}

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

    # Compute second-order blanket with the method of choice.
    blanket = minimize_surface(lb=lb, ub=ub, mode=mode)

    return blanket
