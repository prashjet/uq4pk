
import cvxpy as cp
import numpy as np
from typing import Literal

from .optimizer import Optimizer
from .socp import SOCP


class ECOS(Optimizer):
    """
    Solves SOCP problems using ECOS (via the cvxopt interface).
    """
    def __init__(self, eps: float):
        self.eps = eps

    def setup_problem(self, socp: SOCP, ctol: float, mode: Literal["min", "max"]):
        x = cp.Variable(socp.n)
        sqrt_e = np.sqrt(socp.e)
        constraints = [cp.SOC(sqrt_e, (socp.c @ x - socp.d))]
        # add equality constraint
        if socp.equality_constrained:
            constraints += [socp.a @ x == socp.b]
        if socp.bound_constrained:
            # Cvxpy cannot deal with infinite values. Hence, we have to translate the vector bound x >= lb
            # to the element-wise bound x[i] >= lb[i] for all i where lb[i] > - infinity
            lb = socp.lb
            bounded_indices = np.where(lb > -np.inf)[0]
            if bounded_indices.size > 0:
                constraints += [x[bounded_indices] >= lb[bounded_indices]]
        w = cp.Parameter(socp.n)
        if mode == "min":
            cp_problem = cp.Problem(cp.Minimize(w.T @ x), constraints)
        else:
            cp_problem = cp.Problem(cp.Maximize(w.T @ x), constraints)
        self._cp_problem = cp_problem
        self._x = x
        self._w = w

    def change_loss(self, w: np.ndarray):
        self._w.value = w

    def optimize(self) -> float:
        self._cp_problem.solve(warm_start=True, verbose=False, solver=cp.ECOS)
        optimizer = self._x.value
        return optimizer
