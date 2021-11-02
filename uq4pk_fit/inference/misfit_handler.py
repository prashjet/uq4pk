
from numpy.typing import ArrayLike

from .forward_operator import ForwardOperator
from .parameter_map import ParameterMap


class MisfitHandler:
    """
    handles the misfit, lol
    """
    def __init__(self, y: ArrayLike, op: ForwardOperator, parameter_map: ParameterMap):
        """
        :param y: array_like, shape (M,)
            the noisy measurement
        :param op: ForwardOperator
            The forward operator.
        """
        self._y = y
        self._op = op
        self._pmap = parameter_map
        # It is important that this stays a valid pointer. 'parameter_map' will be changed after initialization of
        # this instance of MisfitHandler.

    def misfit(self, *args):
        f, theta_v = self._pmap.f_theta(list(args))
        return self._op.fwd(f, theta_v) - self._y

    def misfitjac(self, *args):
        f, theta_v = self._pmap.f_theta(list(args))
        full_jac = self._op.jac(f, theta_v)
        jac_x = full_jac[:, self._pmap.full_mask]
        return jac_x
