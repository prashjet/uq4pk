import cvxpy as cp
import numpy as np
import ray

from typing import Sequence
from ..evaluation import AffineEvaluationFunctional
from ..optimization import SOCP


@ray.remote
def solve_remote(socp: SOCP, aef_list_list: Sequence[Sequence[AffineEvaluationFunctional]],
                 actor: ray.actor.ActorHandle, mode: str):
    """
    Remote handle for socp_solve, allowing parallelization via Ray.

    :returns: Of shape (k, n).
        An array where each
    """
    # Create cp-problem
    x = cp.Variable(socp.n)
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
    w = cp.Parameter(socp.n)
    if mode == "min":
        cp_problem = cp.Problem(cp.Minimize(w.T @ x), constraints)
    else:
        cp_problem = cp.Problem(cp.Maximize(w.T @ x), constraints)
    # Initialize out array.
    out_list_list = []
    for aef_pixel_list in aef_list_list:
        out_list = []
        for aef in aef_pixel_list:
            w.value = aef.w
            # First minimize.
            cp_problem.solve(warm_start=True, verbose=False, solver=cp.SCS)
            actor.update.remote(1)
            optimizer = x.value
            out_val = aef.w @ optimizer
            out_list.append(out_val)
        out_list_list.append(out_list)
    out_array = np.array(out_list_list).T
    return out_array


def solve_local(socp: SOCP, aef_list_list: Sequence[Sequence[AffineEvaluationFunctional]], mode: str):
    """
    Remote handle for socp_solve, allowing parallelization via Ray.
    """
    # Create cp-problem
    x = cp.Variable(socp.n)
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
    w = cp.Parameter(socp.n)
    if mode == "min":
        cp_problem = cp.Problem(cp.Minimize(w.T @ x), constraints)
    else:
        cp_problem = cp.Problem(cp.Maximize(w.T @ x), constraints)
    # Initialize out array.
    out_list_list = []
    i = 0
    num_pixels = len(aef_list_list)
    num_scales = len(aef_list_list[0])
    num_problems = num_pixels * num_scales
    for aef_pixel_list in aef_list_list:
        out_list = []
        for aef in aef_pixel_list:
            w.value = aef.w
            # First minimize.
            cp_problem.solve(warm_start=True, verbose=False, solver=cp.SCS)
            print("\r", end="")
            print(f"Solving optimization problem {i+1}/{num_problems}", end=" ")
            optimizer = x.value
            out_val = aef.w @ optimizer
            out_list.append(out_val)
            i += 1
        out_list_list.append(out_list)
    out_array = np.array(out_list_list).T
    return out_array