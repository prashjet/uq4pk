
import numpy as np
from typing import List, Literal

from uq4pk_fit.cgn.problem.parameter import Parameter


class Constraint:
    """
    Represents an abstract constraint.
    """
    def __init__(self, parameters: List[Parameter], fun: callable, jac: callable, ctype: Literal["eq", "ineq"]):
        """

        :param parameters: A list of the parameters involved in the constraint. If the list contains more than one
            element, the constraint will be defined with respect to the concatenated parameter vector.
        :param fun: The function that determines the constraint. Must take ``len(parameters)`` arguments and return
            a numpy array of shape (c,).
        :param jac: The Jacobian of `fun`. Must take arguments of the same form as `fun`, and return a numpy array
            of shape (c, dim), where dim is the dimension of the concatenated parameter vector.
        :param ctype: The type of the constraint.
        """
        # Check input
        self._check_input(parameters, fun, jac, ctype)
        # Compute parameter dimension.
        dim = 0
        for param in parameters:
            dim += param.dim
        self._dim = dim
        self._fun = fun
        self._jac = jac
        # Determine cdim
        testarg = [param.mean for param in parameters]
        y = fun(*testarg)
        self._cdim = y.size
        self._ctype = ctype
        self._parameters = parameters

    def fun(self, *args) -> np.ndarray:
        """
        The constraint function G(x).
        """
        return self._fun(*args)

    def jac(self, *args) -> np.ndarray:
        """
        The constraint jacobian G'(x).
        """
        return self._jac(*args)

    @property
    def ctype(self) -> str:
        """
        The type of the constraint:
            - "eq": equality constraint
            - "ineq": inequality constraint
        """
        return self._ctype

    @property
    def dim(self) -> int:
        """
        The parameter dimension :math:`dim`.
        """
        return self._dim

    @property
    def cdim(self) -> int:
        """
        The dimension :math:`c` of the codomain of the constraint function,
        i.e. :math:`G:\\mathbb{R}^dim \to \\mathbb{R}^c`.
        """
        return self._cdim

    @property
    def parameters(self) -> List[Parameter]:
        """
        The parameters with respect to which the constraint is defined.
        """
        return self._parameters

    @staticmethod
    def _check_input(parameters: List[Parameter], fun: callable, jac: callable, ctype: Literal["eq", "ineq"]):
        if ctype not in ["eq", "ineq"]:
            raise Exception("'ctype' must be either 'eq' or 'ineq'.")
        n = 0
        for param in parameters:
            n += param.dim
        testarg = [param.mean for param in parameters]
        y = fun(*testarg)
        m = y.size
        y_good_shape = y.shape == (m, )
        if not y_good_shape:
            raise Exception(f"The function 'fun' must return numpy arrays of shape ({m}, ).")
        j = jac(*testarg)
        jac_shape_good = j.shape == (m, n)
        if not jac_shape_good:
            raise Exception(f"The function 'jac' must return arrays of shape ({m}, {n}) but return arrays of shape "
                            f"{jac.shape}")