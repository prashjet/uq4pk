
from copy import deepcopy
import numpy as np
from typing import Union

class SOCP:
    """
    Formalizes optimization problems of the form
    min w @ x
    s.t. A x = b
        ||C x - d ||_2^2 <= e
        x >= lb
    """
    def __init__(self, w: np.ndarray, a: Union[np.ndarray, None], b: Union[np.ndarray, None],
                 c: np.ndarray, d: np.ndarray, e: float, lb: Union[np.ndarray, None]):
        # check input
        self._check_input(w, a, b, c, d, e, lb)
        self.w = deepcopy(w)
        self.n = w.size
        self.a = deepcopy(a)
        self.b = deepcopy(b)
        if a is not None:
            self.equality_constrained = True
        else:
            self.equality_constrained = False
        self.c = deepcopy(c)
        self.d = deepcopy(d)
        self.e = deepcopy(e)
        self.lb = deepcopy(lb)
        if self.lb is not None:
            self.bound_constrained = True
        else:
            self.bound_constrained = False

    def check_constraints(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Checks that the vector x satisfies all SOCP-constraints
        :param x:
        :return:
        """
        soc_violation = self.e - np.sum(np.square(self.c @ x - self.d))
        eq_violation = 0
        bound_violation = 0
        constraints_satisfied = (soc_violation >= - tol)
        if self.equality_constrained:
            eq_violation = np.linalg.norm(self.a @ x - self.b, ord=1)
            if eq_violation > tol:
                constraints_satisfied = False
        if self.bound_constrained:
            bound_violation = np.linalg.norm(x - self.lb, ord=1)
            if bound_violation > tol:
                constraints_satisfied = False
        if not constraints_satisfied:
            print(f"Some constraints were violated (tol = {tol})")
            print(f"Violation of SOC constraint: {soc_violation}")
            print(f"Violation of equality constraint: {eq_violation}")
            print(f"Violation of bound constraint: {bound_violation}")
        return constraints_satisfied

    @staticmethod
    def _check_input(w, a, b, c, d, e, lb):
        n = w.size
        assert w.shape == (n, )
        if a is not None:
            n_a = a.shape[0]
            assert a.shape == (n_a, n)
            assert b.shape == (n_a, )
        if c is not None:
            n_c = c.shape[0]
            assert c.shape == (n_c, n)
            assert d.shape == (n_c, )
            assert e is not None
        if lb is not None:
            assert lb.shape == (n, )
