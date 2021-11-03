
import numpy as np

from .socp import SOCP
from .optimizer import Optimizer

import ray


@ray.remote
def socp_solve_remote(socp: SOCP, start: np.ndarray, optimizer: Optimizer):
    x_opt = optimizer.optimize(problem=socp, start=start)
    phi = socp.w @ x_opt
    return phi

def socp_solve(socp: SOCP, start: np.ndarray, optimizer: Optimizer):
    x_opt = optimizer.optimize(problem=socp, start=start)
    phi = socp.w @ x_opt
    return phi