
import numpy as np

from uq4pk_fit.tests.test_cgn.acceptance.test_osborne import OsborneProblem
from uq4pk_fit.tests.test_cgn.acceptance.do_test import do_test


class BoundedOsborneProblem(OsborneProblem):
    def __init__(self):
        OsborneProblem.__init__(self)
        # add nonnegativity constraints
        n = self._problem.n
        lb = np.zeros(n)
        self._problem.set_lower_bound(lb=lb, i=0)
        # rest is same ...


def test_bounded_osborne():
    bop = BoundedOsborneProblem()
    do_test(bop)
