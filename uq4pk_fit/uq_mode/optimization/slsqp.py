"""
Wrapper for Scipy's SLSQP method.
"""

import numpy as np
import scipy.optimize as sciopt

from .constraint import NonlinearConstraint, NullConstraint
from .optimizer import Optimizer
from .optimization_problem import OptimizationProblem
from .socp import SOCP


MAXITER = 500   # if you set this too low, you provoke an "unable to satisfy constraints" error.


class SLSQP(Optimizer):
    """
    Solves the optimization problem arising in the computation of the localized credible intervals.
    """
    def __init__(self, ftol: float = 1e-6, ctol: float = 1e-6):
        self._ftol = ftol
        self._ctol = ctol

    def optimize(self, problem: SOCP, start: np.ndarray) -> np.ndarray:
        """
        Solves optimization problems with different methods
        :param problem: Optimization_Problem
        :return: array_like
            The first output is the minimizer of 'problem'. The second output is a Boolean which indicates whether the
            minimizer satisfies the constraints up to 'ctol'.
        :raises RuntimeError: If the starting point does not satisfy all constraints.
        """
        # Translate SOCP to OptimizationProblem
        optprob = self._translate(problem)
        # Check that the starting value satisfies constraints
        start_feasible, errormsg = optprob.check_constraints(start, self._ctol)
        if not start_feasible:
            raise RuntimeError("The starting point does not satisfy the constraints." + "\n" + errormsg)
        # solve problem with scipy.optimize.minimize:
        problem_solution = sciopt.minimize(method="SLSQP",
                                           fun=optprob.loss_fun,
                                           jac=optprob.loss_grad,
                                           x0=start,
                                           constraints=optprob.constraints,
                                           bounds=optprob.bnds,
                                           options={"maxiter": MAXITER, "ftol": self._ftol})
        # get minimizer
        minimizer = problem_solution.x
        return minimizer

    def _translate(self, problem: SOCP) -> OptimizationProblem:
        """
        Translates an SOCP to an equivalent "OptimizationProblem" object.
        """
        # setup loss function
        if problem.minmax == 0:
            w = problem.w.copy()
        else:
            w = - problem.w.copy()
        def loss_fun(x):
            return w @ x
        def loss_grad(x):
            return w
        # Setup inequality constraint (always present)
        def incon_fun(x):
            return problem.e - np.sum(np.square(problem.c @ x - problem.d))
        def incon_jac(x):
            return - 2 * problem.c.T @ (problem.c @ x - problem.d)
        incon = NonlinearConstraint(fun=incon_fun, jac=incon_jac, type="ineq")
        # Setup equality constraint (Null if not active).
        if problem.equality_constrained:
            def eqcon_fun(x):
                return problem.a @ x - problem.b
            def eqcon_jac(x):
                return problem.a
            eqcon = NonlinearConstraint(fun=eqcon_fun, jac=eqcon_jac, type="eq")
        else:
            eqcon = NullConstraint()
        # Setup bounds
        lb = problem.lb
        # Create and return the OptimizationProblem instance.
        optprob = OptimizationProblem(loss_fun=loss_fun, loss_grad=loss_grad, eqcon=eqcon, incon=incon, lb=lb)
        return optprob
