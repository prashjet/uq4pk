
import cvxpy as cp
import numpy as np

from .optimizer import Optimizer
from .socp import SOCP


class ECOS(Optimizer):
    """
    Solves SOCP problems using ECOS (via the cvxopt interface).
    """
    def __init__(self, scale: float = 1.):
        self._scale = scale

    def optimize(self, problem: SOCP, start: np.ndarray) -> np.ndarray:
        # define the cvxpy program
        cp_problem, x = self._make_cp_problem(problem)
        # Set starting value
        x.value = start
        # Solve
        cp_problem.solve(warm_start=True, verbose=False, solver=cp.ECOS)
        x_opt = x.value
        # return value at optimum or raise exception
        if x_opt is None:
            raise Exception("Encountered infeasible optimization problem.")
        return x_opt

    def _make_cp_problem(self, socp: SOCP) -> cp.Problem:
        # define the optimization vector
        x = cp.Variable(socp.n)
        # add SCOP constraint (||C x - d||_2 <= sqrt(e) <=> ||C x - d||_2^2 <= e)
        constraints = [cp.SOC(np.sqrt(socp.e) / self._scale, (socp.c @ x - socp.d) / self._scale)]
        # add equality constraint
        if socp.equality_constrained:
            constraints += [socp.a @ x - socp.d]
        if socp.bound_constrained:
            constraints += [x >= socp.lb]
        if socp.minmax == 0:
            w = socp.w.copy()
        else:
            w = - socp.w.copy()
        problem = cp.Problem(cp.Minimize(w.T @ x), constraints)
        return problem, x

