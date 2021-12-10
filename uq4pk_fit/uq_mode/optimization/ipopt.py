
import numpy as np
from cyipopt import minimize_ipopt

from .optimization_problem import OptimizationProblem

MAXITER = 3000   # if you set this too low, you provoke an "unable to satisfy constraints" error.
counter = 1


def ipopt(problem: OptimizationProblem, start: np.ndarray, ftol: float = 1e-6, ctol: float = 1e-6) -> np.ndarray:
    """
    Solves an optimization problem using the IPOPT and returns the minimizer.

    :param problem: The optimization problem.
    :param start: The starting point for the optimization. Must satisfy all constraints.
    :param ftol: The desired accuracy of the minimum.
    :param ctol: The tolerance for the constraint satisfaction.
    :return: The minimizer of the optimization problem.
    """
    # Check that the starting value satisfies constraints
    start_feasible, errormsg = problem.check_constraints(start, ctol)
    if not start_feasible:
        raise RuntimeError("The starting point does not satisfy the constraints." + "\n" + errormsg)
    # solve problem with scipy.optimize.minimize:
    # Have to adjust the bounds.
    bnds = np.column_stack([problem.bnds.lb, problem.bnds.ub])
    problem_solution = minimize_ipopt(fun=problem.loss_fun, jac=problem.loss_grad, x0=start,
                                      constraints=problem.constraints, bounds=bnds,
                                      options={"maxiter": MAXITER, "tol": ftol})
    # get minimizer
    minimizer = problem_solution.x
    return minimizer
