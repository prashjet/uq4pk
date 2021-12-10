
import numpy as np

from uq4pk_fit.uq_mode.optimization import ECOS, SLSQP, SOCP, socp_solve
from uq4pk_fit.tests.test_uq_mode.test_optimization.socp_fixture import socp_fixture


def test_socp_solve(socp_fixture):
    socp = socp_fixture[0]
    x0 = socp_fixture[1]
    slsqp = SLSQP()
    ecos = ECOS()
    x_slsqp = slsqp.optimize(problem=socp, start=x0, mode="min")
    x_ecos = ecos.optimize(problem=socp, start=x0, mode="min")
    assert np.isclose(x_slsqp, x_ecos, rtol=1e-2).all()