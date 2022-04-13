import numpy as np
from typing import List

from ..cgn import CGN, DiagonalOperator, Problem, Parameter, LinearConstraint
from ..special_operators import OrnsteinUhlenbeck

from .parameter_map import ParameterMap
from .misfit_handler import MisfitHandler
from .forward_operator import ForwardOperator
from .fitted_model import FittedModel
from .light_weighted_forward_operator import LightWeightedForwardOperator


class LightWeightedModel:
    """
    Abstract base class for that manages the optimization problem, regularization, and optionally also the
    uncertainty quantification.
    The full statistical inference is
    y ~ fwd(f, theta_v) + noise,
    f ~ normal(f_bar, cov1), where cov1 = (beta1 * P1 @ P1.T)^(-1),
    theta_v ~ normal(theta_v_bar, cov2), where
    f >= 0.
    theta_v ~ normal(theta_bar, cov2), where cov2 is determined through P2 and beta2.
    noise ~ normal(0, standard_deviation**2 * identity).
    The user has also the option to fix theta_v via the methods 'fix_theta_v_partially' and 'fix_theta_v'.
    """
    def __init__(self, y: np.array, y_sd: np.array, forward_operator: ForwardOperator, theta: np.array):
        """
        :param y: Of shape (m,). The measurement data.
        :param y_sd: Of shape (m,) If provided, gives a vector of standard deviations for y.
        :param forward_operator: ForwardOperator
        """
        # setup misfit handler
        self._op = LightWeightedForwardOperator(forward_operator=forward_operator, y=y, theta=theta)
        self.weights = self._op.weights

        self.y = y
        self._R = DiagonalOperator(dim=y.size, s=1 / y_sd)
        # get parameter dimensions from misfit handler
        self.m_f = forward_operator.m_f
        self.n_f = forward_operator.n_f
        self.dim_f = self.m_f * self.n_f
        self.dim_y = y.size
        self.snr = np.linalg.norm(y) / np.linalg.norm(y_sd)
        self.dim_theta = forward_operator.dim_theta
        self.theta = theta
        # initialize parameter map
        self._parameter_map = ParameterMap(dim_f=self.dim_f, dim_theta=self.dim_theta)

        # Fix theta by default.
        self._parameter_map.fix_theta(np.arange(self.dim_theta), theta)

        # initialize misfit handler
        self._misfit_handler = MisfitHandler(y=y, op=self._op, parameter_map=self._parameter_map)

        # SET DEFAULT PARAMETERS
        self.lb_f = np.zeros(self.dim_f)
        self._scale = self.dim_y

        # SET DEFAULT REGULARIZATION PARAMETERS
        # set regularization parameters for f
        self.beta1 = self.snr * 1e3     # rule of thumb found after hours of trial-and-error
        self.f_bar = np.zeros(self.dim_f)
        h = np.array([4, 2])
        self.P1 = OrnsteinUhlenbeck(m=self.m_f, n=self.n_f, h=h)

        # SET DEFAULT STARTING VALUES FOR OPTIMIZATION
        # default starting values
        self.f_start = np.ones(self.dim_f) / self.dim_f / self.weights
        self._eqcon = None

        # Initialize solver
        self._solver = self._setup_solver()

    def normalize(self, mass: float):
        """
        Turns normalization on.
        """
        a = np.ones((1, self.dim_f))
        b = mass * np.ones((1,))
        self._set_equality_constraint_for_f(a, b)

    def fit(self):
        """
        Runs the optimization to fit the inference.
        :return: FittedModel
        """
        # Create CGN problem
        problem = self._setup_problem()
        # Translate starting values
        x_start = self._parameter_map.x(self.f_start, self.theta)
        solution = self._solver.solve(problem=problem, starting_values=x_start)
        x_map = [solution.minimizer("f")]
        if not self._parameter_map.theta_fixed:
            x_map.append(solution.minimizer("theta"))
        if not solution.success:
            print("WARNING: Optimization did not converge.")
        # assemble the FittedModel object
        fitted_model = self._assemble_fitted_model(x_map=x_map, problem=problem)
        return fitted_model

    def f_and_theta_v(self, x):
        """
        Given the parameter list, returns f and theta_v
        :param List[numpy.ArrayLike] x:
        :returns: f, theta_v
        """
        f, theta_v = self._parameter_map.f_theta(x)
        return f, theta_v

    # PROTECTED

    def _assemble_fitted_model(self, x_map, problem) -> FittedModel:
        x_start = self._parameter_map.x(self.f_start, self.theta)
        fitted_model = FittedModel(x_map=x_map, problem=problem, parameter_map=self._parameter_map, m_f=self.m_f,
                                   n_f=self.n_f, dim_theta=self.dim_theta, starting_values=x_start)
        return fitted_model

    def _set_equality_constraint_for_f(self, a, b):
        """
        Defines a linear equality constraint for f of the form A @ f = b
        :param a: (dim_y, dim_f) array
        :param b: (dim_y,) vector
        """
        self._eqcon = {"a": a, "b": b}

    def _setup_problem(self):
        # Create parameter objects
        f = Parameter(dim=self._parameter_map.dims[0], name="f")
        f.mean = self.f_bar
        f.beta = self.beta1
        f.regop = self.P1
        f.lb = self.lb_f
        parameters = [f]
        # Add equality constraints
        if self._eqcon is not None:
            constraints = [LinearConstraint(parameters=[f], a=self._eqcon["a"], b=self._eqcon["b"], ctype="eq")]
        else:
            constraints = None
        # create an object of type cgn.Problem
        problem = Problem(parameters=parameters,
                          fun=self._misfit_handler.misfit,
                          jac=self._misfit_handler.misfitjac,
                          q=self._R,
                          constraints=constraints,
                          scale=self._scale)
        return problem

    @staticmethod
    def _setup_solver():
        solver = CGN()
        # Set default solver options
        solver.options.maxiter = 500
        solver.options.timeout = 30
        solver.options.set_verbosity(lvl=2)
        solver.options.tol = 1e-5
        solver.options.ctol = 1e-6
        return solver