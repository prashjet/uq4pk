from uq_mode.external_packages import cgn

from tests.testutils import *


def get_problem() -> TestProblem:
    # SETUP PROBLEM
    a = np.array( [[1., 2.], [3., 4]] )
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
    cgn_problem = cgn.Problem(dims=[2], fun=misfit, jac=misfitjac, q=cgn.DiagonalOperator(dim=2, s=1 / sigma))
    cgn_problem.set_regularization(i=0, beta=1.)
    gauss_newton = cgn.CGN()
    # COMPUTE SOLUTION
    gauss_newton.options.set_verbosity(lvl=2)
    ggn_sol = gauss_newton.solve(cgn_problem, starting_values=[np.zeros(2)])
    x_map = ggn_sol.minimizer[0]
    prec_hat = ggn_sol.precision
    # CHECK MAP AND COVARIANCE ESTIMATES
    prec_post = a.T @ a / (sigma**2) + np.identity(2)
    w_post = np.linalg.solve(prec_post, a.T @ y /(sigma**2))
    x_post = x_bar + w_post
    print(f"Posterior mean error: {np.linalg.norm(x_map-x_post)}")
    print(f"Posterior precision error: {np.linalg.norm(prec_hat - prec_post)}")
    # Build linear model
    model = uq_mode.LinearModel(h = a, y=y, q=np.eye(2) / sigma, m=x_bar, r=np.eye(2), a=None, b=None, lb=None)

    test_problem = TestProblem(x_map=x_map, x_true=x_true, model=model, optimization_problem=cgn_problem)
    return test_problem