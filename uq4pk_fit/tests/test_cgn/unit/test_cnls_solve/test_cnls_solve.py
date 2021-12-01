
import numpy as np
import pytest

from uq4pk_fit.cgn.regop import IdentityOperator
from uq4pk_fit.cgn.cnls_solve import CNLS, cnls_solve
from uq4pk_fit.cgn.cnls_solve.cnls_constraint import ConcreteConstraint
from uq4pk_fit.cgn.cnls_solve.solveroptions import Solveroptions
from uq4pk_fit.cgn.cnls_solve.linesearch_options import LinesearchOptions


def test_cnls_solve():
    m = 10
    n = 20
    i_n = np.eye(n)
    h = i_n[:m, :]
    x = np.arange(n)
    mean = np.zeros(n)
    regop = IdentityOperator(dim=n)

    def fun(x):
        return h @ x

    def jac(x):
        return h

    lb = np.zeros(n)
    ub = np.inf * np.ones(n)
    a = np.ones((1, n))
    b = np.sum(x) * np.ones((1,))
    c = a.copy()
    d = b - 1
    # make CNLS
    eqcon = ConcreteConstraint(dim=n, a=a, b=b)
    incon = ConcreteConstraint(dim=n, a=c, b=d)
    cnls = CNLS(func=fun, jac=jac, q=IdentityOperator(dim=n), m=mean, r=regop, eqcon=eqcon, incon=incon, lb=lb, ub=ub,
                scale=1.)
    # solve
    options = Solveroptions()
    options.set_verbosity(0)
    options.set_verbosity(-1)
    assert options.verbosity == 0
    loptions = LinesearchOptions()
    solution = cnls_solve(cnls=cnls, start=x, options=options, linesearch_options=loptions)
    x_min = solution.minimizer
    # the solution should at least satisfy all constraints
    assert np.isclose(a @ x_min, b).all()
    assert np.all(c @ x_min >= d)
    assert np.all(x_min >= lb)
    # check that error is thrown if start is meaningless
    bad_start = np.ones(3)
    with pytest.raises(ValueError) as e_info:
        bad_solution = cnls_solve(cnls, bad_start, options, loptions)

