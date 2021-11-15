
import numpy as np

from uq4pk_fit.uq_mode.optimization import SLSQP
from uq4pk_fit.tests.test_uq_mode.test_optimization.socp_fixture import socp_fixture


def test_translate(socp_fixture):
    socp = socp_fixture[0]
    slsqp = SLSQP()
    optprob = slsqp._translate(socp)
    assert np.isclose(optprob.lb, socp.lb).all()
    x_test = np.random.randn(socp.w.size)
    assert np.isclose(socp.w @ x_test, optprob.loss_fun(x_test))
    assert np.isclose(socp.w, optprob.loss_grad(x_test)).all()
    incon_ref = socp.e - np.sum(np.square(socp.c @ x_test - socp.d))
    assert np.isclose(incon_ref, optprob.incon.fun(x_test))
    inconjac_ref = - 2 * socp.c.T @ (socp.c @ x_test - socp.d)
    assert np.isclose(inconjac_ref, optprob.incon.jac(x_test)).all()
    eqcon_ref = socp.a @ x_test - socp.b
    assert np.isclose(optprob.eqcon.fun(x_test), eqcon_ref).all()



