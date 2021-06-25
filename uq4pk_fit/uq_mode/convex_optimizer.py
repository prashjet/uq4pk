

import numpy as np
import scipy.optimize as sciopt


class ConvexOptimizer:
    """
    OptimizationWrapper for optimization problems.
    """

    def optimize(self, objective, constraint, lb, x0, conservative=False):
        lossfun = objective["fun"]
        lossgrad = objective["grad"]
        confun = constraint["fun"]
        conjac = constraint["jac"]
        con = {"type": "ineq", "fun": confun, "jac": conjac}
        ub = np.inf * np.ones(lb.size)
        bnds = sciopt.Bounds(lb=lb, ub=ub)
        if conservative:
            # if the optimizer struggles with the constraints, the constraints are set to equality
            eqcon = {"type": "eq", "fun": confun, "jac": conjac}
            def obj_hessian(x):
                return np.zeros((x0.size, x0.size))
            # deactive bounds so to be not confused
            bnds = None
            sol = sciopt.minimize(method='trust-constr', fun=lossfun, jac=lossgrad, hess=obj_hessian, x0=x0,
                                  constraints=eqcon,
                                   bounds=bnds, options={"verbose": 1})
            xopt_unbounded = sol.x
            # reactivate bounds by cutting off
            xopt = xopt_unbounded.clip(min=lb, max=ub)
        else:
            xopt = sciopt.minimize(method='SLSQP', fun=lossfun, jac=lossgrad, x0=x0, constraints=con, bounds=bnds).x
        return xopt



