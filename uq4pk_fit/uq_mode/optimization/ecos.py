
import cvxpy as cp
import numpy as np
from typing import Literal

from .optimizer import Optimizer
from .socp import SOCP


class ECOS(Optimizer):
    """
    Solves SOCP problems using ECOS (via the cvxopt interface).
    """
    def __init__(self, abstol: float = 1e-8):
        self._abstol = abstol

    def optimize(self, problem: SOCP, start: np.ndarray, ctol: float, mode: Literal["min", "max"]) -> np.ndarray:
        # define the cvxpy program
        cp_problem, x = self._make_cp_problem(problem, mode)
        # Set starting value
        x.value = start
        # Solve
        cp_problem.solve(warm_start=True, verbose=False, solver=cp.ECOS, abstol=self._abstol)
        x_opt = x.value
        constraints_satisfied, errormsg = problem.check_constraints(x_opt, ctol)
        if not constraints_satisfied:
            print(errormsg)
        # return value at optimum or raise exception
        if x_opt is None:
            raise Exception("Encountered infeasible optimization problem.")
        return x_opt

    def _make_cp_problem(self, socp: SOCP, mode: Literal["min", "max"]):
        # define the optimization vector
        x = cp.Variable(socp.n)
        # add SCOP constraint (||C x - d||_2 <= sqrt(e) <=> ||C x - d||_2^2 <= e)
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
        w = socp.w.copy()
        if mode == "min":
            problem = cp.Problem(cp.Minimize(w.T @ x), constraints)
        else:
            problem = cp.Problem(cp.Maximize(w.T @ x), constraints)
        return problem, x
