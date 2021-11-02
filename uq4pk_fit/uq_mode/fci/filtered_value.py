"""
Contains class
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Union

from ..filter import Filter
from ..optimization.constraint import Constraint, NonlinearConstraint, NullConstraint


class FilterValue:
    """
    Represents the filtered value.
    phi(x) = filter.evaluate(x).
    """

    def __init__(self, x_map: ArrayLike, filter: Filter):
        self._xmap = x_map
        self._indices = filter.indices
        self._weights = filter.weights
        # Since x(z)_i = x_i if i in window and x(z)_i = xmap otherwise, the Jacobian of the map z -> x(z) is
        # an (n,l) matrix (where l is the window size), where for the j-th index in window the window[j]-th row is set
        # equal to the j-th row of the lxl-identity matrix, and all other rows are equal zero.
        self._l = filter.size
        self._n = x_map.size
        self._x_jac = np.zeros((self._n, self._l))
        id_l = np.identity(self._l)
        self._x_jac[self._indices, :] = id_l[:, :]
        # Define the gradient of the loss function (with respect to z). Since the loss function with respect to z can be
        # written as phi(z) = sum_i w_i z_i, the gradient is just the "weights" vector.
        self._lossgrad = self._weights

    def phi(self, z):
        """.
        :param z: (l,) array
        :return: float
        """
        phi = self._weights @ z
        return phi

    def phi_grad(self, z):
        """
        The gradient of "loss_fun".
        :param z: (l,) array
        :return: (l,) array
        """
        nabla_phi = self._lossgrad
        return nabla_phi

    def phi_hess(self, z):
        """
        The Hessian of "loss_fun".
        :param z:
        :return:
        """
        return np.zeros((self._l, self._l))

    def x(self, z):
        """
        The free variable z is just the value to "xmap" in the localization window. Thus, x is obtained from z
        by adding z to xmap for all the frame coordinates.
        :param z: (l,) array
        :return: (n,) array
            Here, "n" denotes the dimension of the full parameter space.
        """
        x_z = self._xmap.copy()
        x_z[self._indices] = z
        return x_z

    @property
    def dx_dz(self):
        """
        Returns the Jacobian of the function "x".
        :param z: (l,) array
        :return: (n,l) array
        """
        return self._x_jac

    @property
    def initial_value(self):
        """
        The initial value for z is x_map[window].
        :return: (l,) numpy array of floats
        """
        return self._xmap[self._indices]

    def transform_linear_constraint(self, a: Union[ArrayLike, None], b: Union[ArrayLike, None], type: str) -> Constraint:
        """
        Transforms an equality constraint of the form a @ x = b
        to z-space.
        :returns: dict or None
            - fun_z: callable
            - jac_z: callable
        """
        if a is None:
            constraint_z = NullConstraint()
        else:
            c = a.shape[0]
            assert b.shape == (c, )
            # Check if constraint is even active at z
            a_z = a @ self._x_jac
            def fun_z(z):
                return a @ self.x(z) - b
            def jac_z(z):
                return a_z
            constraint_z = NonlinearConstraint(fun=fun_z, jac=jac_z, type=type)
        return constraint_z

    def transform_nonlinear_constraint(self, fun: callable, jac: callable, type: str) -> Constraint:
        """
        Transforms a constraint of the form fun(x) >= 0 (or jac(x) >= 0) to z-space.
        """
        def fun_z(z):
            x = self.x(z)
            y = fun(x)
            return y
        def jac_z(z):
            x = self.x(z)
            j = jac(x) @ self.dx_dz
            return j
        constraint_z = NonlinearConstraint(fun=fun_z, jac=jac_z, type=type)
        return constraint_z

    def lower_bound(self, lb: Union[ArrayLike, None]):
        """
        Translates lower bound to z-space.
        :returns: (l,) array or None
        """
        if lb is None:
            lb_z = None
        else:
            lb_z = lb[self._indices]
            if np.isinf(lb_z).all():
                # if all bounds are inactive, just return None
                lb_z = None
        return lb_z