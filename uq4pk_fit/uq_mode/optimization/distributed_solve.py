import cvxpy as cp
import numpy as np
import ray
from typing import Literal

from typing import Sequence, Optional
from uq4pk_fit.uq_mode.evaluation import AffineEvaluationFunctional
from uq4pk_fit.uq_mode.optimization import SOCP
from ..optimization import Optimizer


SOLVER = cp.SCS     # For larger problems, SCS solver better.


@ray.remote
def solve_distributed_remote(socp: SOCP, aef_list_list: Sequence[Sequence[AffineEvaluationFunctional]],
                             optimizer: Optimizer, mode: Literal["min", "max"], ctol: float,
                             actor: Optional[ray.actor.ActorHandle]):
    """
    Remote handle for socp_solve, allowing parallelization via Ray.

    :returns: Of shape (k, n).
    """
    return solve_distributed(socp, aef_list_list, optimizer, mode, ctol, actor)


def solve_distributed(socp: SOCP, aef_list_list: Sequence[Sequence[AffineEvaluationFunctional]], optimizer: Optimizer,
                      mode: Literal["min", "max"], ctol: float, actor: Optional[ray.actor.ActorHandle] = None):
    """
    Remote handle for socp_solve, allowing parallelization via Ray.

    :returns: Of shape (k, n).
    """
    # Setup optimizer.
    optimizer.setup_problem(socp=socp,  ctol=ctol, mode=mode)
    # Initialize out array.
    out_list_list = []
    for aef_pixel_list in aef_list_list:
        out_list = []
        for aef in aef_pixel_list:
            optimizer.change_loss(w=aef.w)
            z_opt = optimizer.optimize()
            out_val = aef.phi(z_opt)
            if actor is not None:
                actor.update.remote(1)
            out_list.append(out_val)
        out_list_list.append(out_list)
    out_array = np.array(out_list_list).T
    return out_array