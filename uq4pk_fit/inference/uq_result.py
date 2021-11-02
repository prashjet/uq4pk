"""
Container for uq_result.
"""

from typing import Union
from ..uq_mode import FilterFunction


class UQResult:

    def __init__(self, ci_f, filter_f: Union[FilterFunction, None], ci_theta,
                 filter_theta: Union[FilterFunction, None]):
        """
        :param ci_f: None or array_like, shape (N,2)
            The credible intervals for f.
        :param filter_f: uq_mode.FilterFunction
            Must be provided if ci_f is not None.
        :param ci_theta: None or array_like, shape (M, 2)
            The credible intervals for theta.
        """
        # check input for consistency
        self._check_input(ci_f, filter_f)
        self._check_input(ci_theta, filter_theta)
        self.ci_f = ci_f
        self.filter_f = filter_f
        self.ci_theta = ci_theta
        self.filter_theta = filter_theta

    @staticmethod
    def _check_input(ci_f, filter_f):
        if ci_f is not None:
            assert ci_f.shape[0] == filter_f.dim


class NullUQResult(UQResult):
    def __init__(self):
        UQResult.__init__(self, ci_f=None, ci_theta=None, filter_f=None, filter_theta=None)
