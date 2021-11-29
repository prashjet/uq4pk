
import numpy as np
import scipy.optimize as sciopt

from .optimization_problem import OptimizationProblem


MAXITER = 500   # if you set this too low, you provoke an "unable to satisfy constraints" error.
counter = 1

def slsqp(problem: OptimizationProblem, start: np.ndarray, ftol: float = 1e-6, ctol: float = 1e-6) -> np.ndarray:
    """
    Solves an optimization problem using scipy's SLSQP method and returns the minimizer.

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
    problem_solution = sciopt.minimize(method="SLSQP",
                                       fun=problem.loss_fun,
                                       jac=problem.loss_grad,
                                       x0=start,
                                       constraints=problem.constraints,
                                       bounds=problem.bnds,
                                       options={"maxiter": MAXITER, "ftol": ftol})
    # get minimizer
    minimizer = problem_solution.x
    return minimizer