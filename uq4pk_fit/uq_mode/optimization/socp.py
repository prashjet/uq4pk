
from copy import deepcopy
import numpy as np
from typing import Union


class SOCP:
    """
    Formalizes optimization problems of the form
    min / max_w @ x
    s.t. A x = b
        ||C x - d ||_2^2 <= e
        x >= lb
    """
    def __init__(self, w: np.ndarray, a: Union[np.ndarray, None], b: Union[np.ndarray, None],
                 c: np.ndarray, d: np.ndarray, e: float, lb: Union[np.ndarray, None]):
        # check input
        self._check_input(w, a, b, c, d, e, lb)
        # check that the equality constraints satisfy constraint qualification
        self._check_constraint_qualification(a, b)
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

    def cost(self, x):
        return self.w @ x

    def costgrad(self, x):
        return self.w

    def check_constraints(self, x: np.ndarray, tol: float = 1e-10):
        """
        Checks that the vector x satisfies all SOCP-constraints

        :param x:
        :param tol:
        :returns:
            - constraints_satisfied: bool. True if all constraints are satisfied, otherwise False.
            - message: str. An error message specifying which constraints were violated.
        """
        soc_violation = max(0, np.sum(np.square(self.c @ x - self.d)) - self.e) / self.e
        print(f"SOC violation = {soc_violation}.")
        soc_violated = soc_violation > tol
        if self.equality_constrained:
            eqcon_violated = self._l1norm((self.a @ x - self.b)) > tol
        else:
            eqcon_violated = False
        if self.bound_constrained:
            bound_violated = self._l1norm((x - self.lb).clip(max=0.)) > tol
        else:
            bound_violated = False
        constraints_satisfied = not (soc_violated or eqcon_violated or bound_violated)
        message = "The following constraints have been violated: \n"
        if soc_violated:
            message += f"cone condition \n ({soc_violation} / {tol}"
        if eqcon_violated:
            message += "equality constraint \n"
        if bound_violated:
            message += "bound constraint \n"
        return constraints_satisfied, message

    @staticmethod
    def _l1norm(x):
        """
        Had to build my own l1-norm since numpy.linalg.norm(..., ord=1) does not work with floats.
        """
        return np.sum(np.abs(x))

    @staticmethod
    def _check_constraint_qualification(a: Union[np.ndarray, None], b: Union[np.ndarray, None]):
        if a is not None:
            c = a.shape[0]
            if not np.linalg.matrix_rank(a) >= c:
                raise Exception("Exception in definition of SCOP: 'a' does not satisfy constraint qualification.")

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
