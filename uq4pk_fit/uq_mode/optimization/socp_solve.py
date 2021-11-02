
import cvxpy as cp
import numpy as np
import scipy as sp

from .socp import SOCP


class SOCPSolver:
    """
    Solves SOCP problems using cvxopt.
    """

    def solve(self, socp: SOCP, start: np.ndarray, scale: float):
        # define the cvxpy program
        cp_problem, x = self._make_cp_problem(socp, scale)
        # Set starting value
        x.value = start
        # Solve
        cp_problem.solve(warm_start=True, verbose=False, abstol=1e-1)
        x_opt = x.value
        # return value at optimum
        return x_opt

    def _make_cp_problem(self, socp: SOCP, scale: float):
        # define the optimization vector
        x = cp.Variable(socp.n)
        # add SCOP constraint
        constraints = [cp.SOC(np.sqrt(socp.e), socp.c @ x - socp.d)]
        # add equality constraint
        if socp.equality_constrained:
            constraints += [socp.a @ x - socp.d]
        if socp.bound_constrained:
            constraints += [x >= socp.lb]
        problem = cp.Problem(cp.Minimize(socp.w.T @ x), constraints)
        return problem, x

