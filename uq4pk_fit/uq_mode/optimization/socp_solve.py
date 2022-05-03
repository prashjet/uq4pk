
import numpy as np
from time import time
from typing import Literal, Optional, Sequence

from ..evaluation import AffineEvaluationFunctional
from .optimizer import Optimizer
from .socp import SOCP

import ray
from ray.actor import ActorHandle


class SolveResult:
    def __init__(self, value: Sequence[float], time: float):
        self.values = value
        self.time = time


@ray.remote
def socp_solve_remote(aef_list: Sequence[AffineEvaluationFunctional], socp: SOCP, mode: Literal["min", "max"],
               optimizer: Optimizer, ctol: float, actor: Optional[ActorHandle] = None) -> SolveResult:
    """
    Remote handle for socp_solve, allowing parallelization via Ray.
    """
    # Check that starting value satisfies SOCP constraints
    return socp_solve(aef_list, socp, mode, optimizer, ctol, actor)


def socp_solve(aef_list: Sequence[AffineEvaluationFunctional], socp: SOCP, mode: Literal["min", "max"],
               optimizer: Optimizer, ctol: float, actor: Optional[ActorHandle] = None) -> SolveResult:
    """
    Remote handle for socp_solve, allowing parallelization via Ray.
    """
    # Check that starting value satisfies SOCP constraints
    t0 = time()
    optimizer.setup_problem(socp=socp, ctol=ctol, mode=mode)
    out_list = []
    for aef in aef_list:
        optimizer.change_loss(w=aef.w)
        z_opt = optimizer.optimize()
        out_val = aef.phi(z_opt)
        if actor is not None:
            actor.update.remote(1)
        out_list.append(out_val)
    t1 = time()
    t_avg = (t1 - t0) / len(aef_list)
    return SolveResult(out_list, t_avg)
