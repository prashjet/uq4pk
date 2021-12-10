import numpy as np
from typing import Literal

from .constraint import NonlinearConstraint, NullConstraint
from .optimizer import Optimizer
from .optimization_problem import OptimizationProblem
from .socp import SOCP
from .ipopt import ipopt


MAXITER = 500   # if you set this too low, you provoke an "unable to satisfy constraints" error.


class IPOPT(Optimizer):
    """
    Solves the optimization problem arising in the computation of the localized credible intervals.
    """
    def __init__(self, ftol: float = 1e-6, ctol: float = 1e-6):
        self._ftol = ftol
        self._ctol = ctol

    def optimize(self, problem: SOCP, start: np.ndarray, mode: Literal["min", "max"]) -> np.ndarray:
        """
        Solves optimization problems with different methods
        :param problem: Optimization_Problem
        :param start: The initial guess.
        :param mode: Determines whether to minimize or to maximize.
        :return: array_like
            The first output is the minimizer of 'problem'. The second output is a Boolean which indicates whether the
            minimizer satisfies the constraints up to 'ctol'.
        :raises RuntimeError: If the starting point does not satisfy all constraints.
        """
        # Translate SOCP to OptimizationProblem
        optprob = self._translate(problem, mode)
        # Solve optimization problem with SLSQP
        minimizer = ipopt(optprob, start)
        return minimizer

    @staticmethod
    def _translate(problem: SOCP, mode: Literal["min", "max"]) -> OptimizationProblem:
        """
        Translates an SOCP to an equivalent "OptimizationProblem" object.
        """
        # setup loss function
        if mode == "min":
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
