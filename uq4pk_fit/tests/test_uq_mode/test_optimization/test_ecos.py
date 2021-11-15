
import numpy as np

from uq4pk_fit.uq_mode.optimization import ECOS
from uq4pk_fit.tests.test_uq_mode.test_optimization.socp_fixture import socp_fixture


def test_make_cp_problem(socp_fixture):
    socp = socp_fixture[0]
    ecos = ECOS()
    cp, x = ecos._make_cp_problem(socp)
    x_test = np.random.randn(socp.w.size)