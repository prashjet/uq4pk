"""
Wrapper for Scipy's SLSQP method.
"""

import scipy.optimize as sciopt
from .optimization_problem import OptimizationProblem


MAXITER = 100


class SLSQP:
    """
    Solves the optimization problem arising in the computation of the localized credible intervals.
    """
    @staticmethod
    def optimize(problem: OptimizationProblem):
        """
        Solves optimization problems with different methods
        :param problem: Optimization_Problem
        :return: array_like
            The first output is the minimizer of 'problem'. The second output is a Boolean which indicates whether the
            minimizer satisfies the constraints up to 'ctol'.
        """
        # solve problem with scipy.optimize.minimize:
        problem_solution = sciopt.minimize(method="SLSQP",
                                           fun=problem.loss_fun,
                                           jac=problem.loss_grad,
                                           x0=problem.start,
                                           constraints=problem.constraints,
                                           bounds=problem.bnds,
                                           options={"maxiter": MAXITER})
        # get minimizer
        minimizer = problem_solution.x
        return minimizer
