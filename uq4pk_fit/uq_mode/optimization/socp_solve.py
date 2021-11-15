
import numpy as np

from .socp import SOCP
from .optimizer import Optimizer

import ray


@ray.remote
def socp_solve_remote(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float, qoi: callable) -> np.ndarray:
    x_opt = optimizer.optimize(problem=socp, start=start)
    # check that minimizer satisfies constraint
    constraints_satisfied, errormsg = socp.check_constraints(x_opt, ctol)
    if not constraints_satisfied:
        print("WARNING: The solver was not able to find a feasible solution." + " " + errormsg)
    return qoi(x_opt)

def socp_solve(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float) -> np.ndarray:
    x_opt = optimizer.optimize(problem=socp, start=start)
    constraints_satisfied, errormsg = socp.check_constraints(x_opt, ctol)
    if not constraints_satisfied:
        raise RuntimeError("The solver was not able to find a feasible solution." + " " + errormsg)
    # Compute return value
    return x_opt