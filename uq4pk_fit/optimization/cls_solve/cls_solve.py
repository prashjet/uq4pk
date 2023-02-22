import numpy as np

from .cls import CLS
import qpsolvers as qp


def cls_solve(cls: CLS) -> np.ndarray:
    """
    Solves a constrained least-squares problem.
    """
    # Bring the CLS problem in the right format.
    r, s, g, h, a, b, lb, ub = _bring_problem_in_right_form(cls)
    # Solve the problem with qpsolvers.
    x_min = qp.solve_ls(R=r, s=s, G=g, h=h, A=a, b=b, lb=lb, ub=ub, solver="quadprog", verbose=True)
    if x_min is None:
        raise RuntimeError("CLS-solver could not find a solution.")
    return x_min


def _bring_problem_in_right_form(cls: CLS):
    """
    Brings the CLS problem in the right format so that qpsolvers.solve_ls can solve it.
    """
    r = cls.h
    s = cls.y
    g = None
    h = None
    a = None
    b = None
    if cls.bound_constrained:
        lb = cls.lb.astype(np.float64)
        ub = cls.ub.astype(np.float64)
    else:
        lb = None
        ub = None
    return r, s, g, h, a, b, lb, ub
