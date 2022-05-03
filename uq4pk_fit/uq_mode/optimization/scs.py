import cvxpy as cp
import numpy as np
from typing import Literal

from .optimizer import Optimizer
from .socp import SOCP


class SCS(Optimizer):
    """
    Solves SOCP problems using SCS (via the cvxopt interface).

    WARNING: SCS really doesn't like it if lb != 0.
    """
    def __init__(self):
        self._wnorm = 0.

    def setup_problem(self, socp: SOCP, ctol: float, mode: Literal["min", "max"]):
        # In order for SCS to be a little bit better conditioned, we have to transform everything to u = x - lb.
        # or equivalently, x = u + lb.
        if socp.bound_constrained:
            bias = socp.lb
        else:
            bias = np.zeros(socp.n)
        u = cp.Variable(socp.n)
        sqrt_e = np.sqrt(socp.e)
        b = np.linalg.norm(socp.c @ bias - socp.d)
        constraints = [cp.SOC(sqrt_e, (socp.c @ u + socp.c @ bias - socp.d))]
        # add equality constraint
        if socp.equality_constrained:
            constraints += [socp.a @ u == socp.b + socp.a @ bias]
        if socp.bound_constrained:
            # Cvxpy cannot deal with infinite values. Hence, we have to translate the vector bound x >= lb
            # to the element-wise bound x[i] >= lb[i] for all i where lb[i] > - infinity
            lb = np.zeros(socp.n)
            bounded_indices = np.where(lb > -np.inf)[0]
            if bounded_indices.size > 0:
                constraints += [u[bounded_indices] >= lb[bounded_indices]]
        w = cp.Parameter(socp.n)
        if mode == "min":
            cp_problem = cp.Problem(cp.Minimize(w.T @ u), constraints)
        else:
            cp_problem = cp.Problem(cp.Maximize(w.T @ u), constraints)
        self._cp_problem = cp_problem
        self._u = u
        self._bias = bias
        self._w = w

    def change_loss(self, w: np.ndarray):
        wnorm = np.linalg.norm(w)
        self._w.value = w / wnorm

    def optimize(self) -> float:
        self._cp_problem.solve(warm_start=True, verbose=False, solver=cp.SCS)
        u_optimizer = self._u.value
        x_optimizer = u_optimizer - self._bias
        return x_optimizer