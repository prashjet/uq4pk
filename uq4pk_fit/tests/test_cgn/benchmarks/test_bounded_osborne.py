
import numpy as np

from uq4pk_fit.tests.test_cgn.benchmarks.test_osborne import OsborneProblem
from uq4pk_fit.tests.test_cgn.benchmarks.do_test import do_test


class BoundedOsborneProblem(OsborneProblem):
    def __init__(self):
        OsborneProblem.__init__(self)
        # add nonnegativity constraints
        n = self._problem.n
        lb = np.zeros(n)
        self._problem.parameter("x").lb = lb
        # rest is same ...


def test_bounded_osborne():
    bop = BoundedOsborneProblem()
    do_test(bop)
