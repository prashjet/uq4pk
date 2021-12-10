
import numpy as np
from typing import Literal

from .socp import SOCP
from .optimizer import Optimizer

import ray
from ray.actor import ActorHandle


@ray.remote
def socp_solve_remote(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float, qoi: callable,
                      actor: ActorHandle, mode: Literal["min", "max"]) -> np.ndarray:
    # Check that starting value satisfies SOCP constraints
    constraints_satisfied, errormsg = socp.check_constraints(start, ctol)
    if not constraints_satisfied:
        raise Exception("The starting point is infeasible.")
    x_opt = optimizer.optimize(problem=socp, start=start, mode=mode)
    # check that minimizer satisfies constraint
    constraints_satisfied, errormsg = socp.check_constraints(x_opt, ctol)
    if not constraints_satisfied:
        print("WARNING: The solver was not able to find a feasible solution." + " " + errormsg)
    actor.update.remote(1)
    return qoi(x_opt)


def socp_solve(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float, qoi: callable,
               mode: Literal["min", "max"]) -> np.ndarray:
    # Check that starting value satisfies SOCP constraints
    constraints_satisfied, errormsg = socp.check_constraints(start, ctol)
    if not constraints_satisfied:
        raise Exception("The starting point is infeasible.")
    x_opt = optimizer.optimize(problem=socp, start=start, mode=mode)
    constraints_satisfied, errormsg = socp.check_constraints(x_opt, ctol)
    if not constraints_satisfied:
        raise RuntimeError("The solver was not able to find a feasible solution." + " " + errormsg)
    # Compute return value
    return qoi(x_opt)
