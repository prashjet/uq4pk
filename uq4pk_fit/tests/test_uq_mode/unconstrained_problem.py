
import uq4pk_fit.cgn as cgn
from uq4pk_fit.tests.test_uq_mode.testutils import *
import uq4pk_fit.uq_mode as uq_mode


def get_problem() -> TestProblem:
    # SETUP PROBLEM
    a = np.array( [[1., 2.], [3., 4]] )
    x = cgn.Parameter(dim=2, name="x")
    x.beta = 1.
    sigma = .1
    x_true = np.random.randn(2)
    #x_true = np.zeros((2,))
    noi = sigma * np.random.randn(2)
    y = a @ x_true + noi
    x_bar = np.zeros(2)
    def misfit(x):
        return (a @ x - y)
    def misfitjac(x):
        return a
    cgn_problem = cgn.Problem(parameters=[x], fun=misfit, jac=misfitjac, q=cgn.DiagonalOperator(dim=2, s=1 / sigma))
    gauss_newton = cgn.CGN()
    # COMPUTE SOLUTION
    gauss_newton.options.set_verbosity(lvl=0)
    ggn_sol = gauss_newton.solve(cgn_problem, starting_values=[np.zeros(2)])
    x_map = ggn_sol.minimizer("x")
    prec_hat = ggn_sol.precision
    # CHECK MAP AND COVARIANCE ESTIMATES
    prec_post = a.T @ a / (sigma**2) + np.identity(2)
    w_post = np.linalg.solve(prec_post, a.T @ y /(sigma**2))
    x_post = x_bar + w_post
    print(f"Posterior mean error: {np.linalg.norm(x_map-x_post)}")
    print(f"Posterior precision error: {np.linalg.norm(prec_hat - prec_post)}")
    # Build linear model
    model = uq_mode.LinearModel(h = a, y=y, q=cgn.DiagonalOperator(dim=2, s=1 / sigma), m=x_bar,
                                r=cgn.IdentityOperator(dim=2), a=None, b=None, lb=None)
    test_problem = TestProblem(x_map=x_map, x_true=x_true, model=model, optimization_problem=cgn_problem)
    return test_problem