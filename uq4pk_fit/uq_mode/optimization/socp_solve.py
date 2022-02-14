
import numpy as np
from typing import Literal, Tuple

from .socp import SOCP
from .optimizer import Optimizer

import ray
from ray.actor import ActorHandle


@ray.remote
def socp_solve_remote(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float, qoi: callable,
                      actor: ActorHandle, mode: Literal["min", "max"]):
    """
    Remote handle for socp_solve, allowing parallelization via Ray.
    """
    # Check that starting value satisfies SOCP constraints
    out_tuple = socp_solve(socp=socp, start=start, optimizer=optimizer, ctol=ctol, qoi=qoi, mode=mode)
    actor.update.remote(1)
    return out_tuple


def socp_solve(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float, qoi: callable,
               mode: Literal["min", "max"]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a SOCP problem using a defined optimizer.

    :returns: qoi, x_opt
        - qoi: The quantity of interest, evaluated at the optimizer x_opt.
        - x_opt: The optimizer of the given SOCP problem.
    """
    # Check that starting value satisfies SOCP constraints
    constraints_satisfied, errormsg = socp.check_constraints(start, ctol)
    if not constraints_satisfied:
        raise Exception("The starting point is infeasible.")
    x_opt = optimizer.optimize(problem=socp, start=start, mode=mode)
    constraints_satisfied, errormsg = socp.check_constraints(x_opt, ctol)
    if not constraints_satisfied:
        raise RuntimeError("The solver was not able to find a feasible solution." + " " + errormsg)
    # Compute return value
    return (qoi(x_opt), x_opt)
