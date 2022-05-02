
import numpy as np
from time import time
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
    out = socp_solve(socp=socp, start=start, optimizer=optimizer, ctol=ctol, qoi=qoi,mode=mode)
    actor.update.remote(1)
    return out


def socp_solve(socp: SOCP, start: np.ndarray, optimizer: Optimizer, ctol: float, qoi: callable,
               mode: Literal["min", "max"]) -> Tuple[np.ndarray, float]:
    """
    Solves a SOCP problem using a defined optimizer.

    :returns: qoi, t
        - qoi: The quantity of interest, evaluated at the optimizer x_opt.
        - t: The required computation time.
    """
    # Check that starting value satisfies SOCP constraints
    t0 = time()
    constraints_satisfied, errormsg = socp.check_constraints(start, ctol)
    if not constraints_satisfied:
        raise Exception("The starting point is infeasible.")
    z_opt = optimizer.optimize(problem=socp, start=start, mode=mode, ctol=ctol)
    t1 = time()
    # Compute return value
    return qoi(z_opt), t1 - t0
