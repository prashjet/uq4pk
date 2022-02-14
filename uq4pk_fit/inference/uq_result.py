"""
Container for uq_result.
"""

from typing import Union

import numpy as np

from ..uq_mode import FilterFunction


class UQResult:

    def __init__(self, lower_f, upper_f, filter_f: Union[FilterFunction, None], lower_theta, upper_theta,
                 filter_theta: Union[FilterFunction, None], scale: float, features: np.ndarray = None):
        """
        :param lower_f: None or image.
        :param upper_f: None of image.
        :param lower_theta: None or vector.
        :param upper_theta: None or vector.
        :param filter_f: uq_mode.FilterFunction
            Must be provided if ci_f is not None.
        :param scale: The scale of the uncertainty quantification.
        """
        # check input for consistency
        self._check_input(lower_f, filter_f)
        self._check_input(upper_f, filter_f)
        self._check_input(lower_theta, filter_theta)
        self._check_input(upper_theta, filter_theta)
        self.lower_f = lower_f
        self.upper_f = upper_f
        self.filter_f = filter_f
        self.lower_theta = lower_theta
        self.upper_theta = upper_theta
        self.filter_theta = filter_theta
        self.scale = scale
        self.features = features

    @staticmethod
    def _check_input(vector, filter):
        if vector is not None:
            assert vector.size == filter.dim


class NullUQResult(UQResult):

    def __init__(self):
        UQResult.__init__(self, lower_f=None, upper_f=None, lower_theta=None, upper_theta=None, filter_f=None,
                          filter_theta=None, scale=0.)
