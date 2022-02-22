"""
Contains class "FittedModel"
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Sequence

import uq4pk_fit.cgn as cgn
from uq4pk_fit.cgn.translator.translator import Translator
import uq4pk_fit.uq_mode as uq_mode
import uq4pk_fit.blob_detection as blob_detection
from .jaccard_distance import mean_jaccard_distance
from .make_filter_function import make_filter_function
from .parameter_map import ParameterMap
from .uq_result import UQResult


class FittedModel:
    """
    Represents a fitted inference, i.e. a Model linearized around the map:
        0 = mf(x_map) + fwd'(x_map)(x-x_map) + noise,
        noise ~ normal(0, identity / delta^2),
        x ~ normal(x_bar, cov), where cov = (P @ P.T)^(-1),
        x >= lb, A @ x = b.
    """
    def __init__(self, x_map: List[ArrayLike], problem: cgn.Problem, parameter_map: ParameterMap, m_f, n_f, dim_theta,
                 starting_values: List[ArrayLike]):
        """
        """
        self._x_map = x_map
        self._x_map_vec = np.concatenate(x_map)
        self._n = problem.n
        self._parameter_map = parameter_map
        self._problem = deepcopy(problem)
        # create linearized model
        self._linearized_model = self._create_linearized_model(problem)
        cost_map = self._problem.costfun(*x_map)
        cost_lin = self._linearized_model.cost(self._x_map_vec)
        assert np.isclose(cost_map, cost_lin)
        # create f_map and theta_map by translating x_map
        self._f_map, self._theta_map = parameter_map.f_theta(x_map)
        self._m_f = m_f
        self._n_f = n_f
        self._dim_theta = dim_theta
        self._dim_f = self._m_f * self._n_f
        self._starting_values = starting_values

    def costfun(self, f, theta_v):
        x = self._parameter_map.x(f, theta_v)
        cost = self._problem.costfun(*x)
        return cost

    @property
    def f_map(self):
        """
        Returns MAP estimate for age-metallicity distribution.
        :return:
        """
        return self._f_map.copy()

    @property
    def theta_map(self):
        """
        Returns MAP estimate for theta_v
        :return:
        """
        return self._theta_map.copy()

    def uq(self, method: str = "fci", options: dict = None) -> UQResult:
        """
        Performs uncertainty quantification using the method of choice.
        :param str="fci" method:
            The method used for uncertainty quantification. Options are:
            - "lci": Local credible intervals computed with the method described in Cui et al.
            - "fci": Filtered credible intervals.
            - "mc": Filtered credible intervals, computed with a heuristic Monte Carlo method.
        :param Optional[dict] options:
            A dict specifying further options. The possible options depend on the method used (...).
        :returns: UQResult
        """
        if options is None:
            options = {}
        if method == "fci":
            return self._uq_fci(options)
        elif method == "mc":
            return self._uq_mc(options)
        elif method == "dummy":
            return self._uq_dummy(options)
        elif method == "lci":
            return self._uq_lci(options)
        else:
            raise KeyError("Unknown method.")


    def _create_linearized_model(self, problem: cgn.Problem) -> uq_mode.LinearModel:
        # Then, compute all entities necessary for building the linearized model from the CNLS problem.
        translator = Translator(problem)
        cnls = translator.translate()
        f_map = cnls.func(self._x_map_vec)
        j_map = cnls.jac(self._x_map_vec)
        q = cnls.q
        r = cnls.r
        if cnls.equality_constrained:
            a = cnls.g_jac(self._x_map_vec)
            b = a @ self._x_map_vec - cnls.g(self._x_map_vec)
        else:
            a = None
            b = None
        x_bar = cnls.m
        lb = cnls.lb
        # Form the LinearizedModel object
        model = uq_mode.LinearModel(h=j_map,
                                    y=-f_map + j_map @ self._x_map_vec,
                                    q=q,
                                    a=a,
                                    b=b,
                                    m=x_bar,
                                    r=r,
                                    lb=lb)
        return model

    def _uq_fci(self, options: dict):
        """
        Computes filtered credible intervals using the Pereyra approximation.
        :param dict options:
        :return: UQResult
        """
        # Create appropriate filter
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        # compute filtered credible intervals
        fci_obj = uq_mode.fci(alpha=0.05, x_map=self._x_map_vec, model=self._linearized_model,
                                                       ffunction=filter_function, options=options)
        ci_x = fci_obj.interval
        ci_f, ci_theta = self._parameter_map.ci_f_theta(ci_x)
        uq_scale = options["h"]
        # Reshape
        lower_f = self._reshape_f(ci_f[:, 0])
        upper_f = self._reshape_f(ci_f[:, 1])
        uq_result = UQResult(lower_f=lower_f, upper_f=upper_f, lower_theta=ci_theta[:, 0], upper_theta=ci_theta[:, 1],
                             filter_f=filter_f, filter_theta=filter_theta, scale=uq_scale)
        if options["detailed"]:
            return uq_result, fci_obj.minimizers, fci_obj.maximizers
        else:
            return uq_result

    def _uq_lci(self, options: dict):
        """
        Computes local credible intervals using the Pereyra approximation.
        Only works for the linear model!

        :param options:
        :return: UQResult
        """
        if not self._parameter_map.theta_fixed:
            raise Exception("Local credible intervals only implemented for the linear model.")
        # Set required parameters
        alpha = 0.05
        # Create appropriate partition
        a = options.setdefault("a", 2)
        b = options.setdefault("b", 2)
        partition = uq_mode.rectangle_partition(m=self._m_f, n=self._n_f, a=a, b=b)
        lci_f = uq_mode.lci(alpha=alpha, model=self._linearized_model, x_map=self._x_map_vec, partition=partition,
                            options=options)
        filter_f = uq_mode.IdentityFilterFunction(dim=self._dim_f)
        lower_f = self._reshape_f(lci_f[:, 0])
        upper_f = self._reshape_f(lci_f[:, 1])
        uq_result = UQResult(lower_f=lower_f, upper_f=upper_f, lower_theta=None, upper_theta=None,
                             filter_f=filter_f, filter_theta=None, scale=1)
        return uq_result

    def _uq_mc(self, options: dict):
        """
        Computes filtered credible intervals using randomized resampling.
        :param dict options:
        :return: UQResult
        """
        alpha = 0.05  # 95%-credibility
        # As remarked in Cappellari and Emsellem, one want to use smaller values for the regularization parameters
        # when performing RML than when performing MAP estimation.
        # Setup the uq_mode.fci.FilterFunction object
        options["reduction"] = options.setdefault("reduction", 42)
        uq_scale = options.setdefault("h", 3)
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        ci_x = uq_mode.fci_rml(alpha=alpha, model=self._linearized_model, x_map=self._x_map_vec,
                               ffunction=filter_function, options=options)
        # Make UQResult object.
        ci_f, ci_theta = self._parameter_map.ci_f_theta(ci_x)
        # Reshape
        lower_f = self._reshape_f(ci_f[:, 0])
        upper_f = self._reshape_f(ci_f[:, 1])
        uq_result = UQResult(lower_f=lower_f, upper_f=upper_f, lower_theta=ci_theta[:, 0], upper_theta=ci_theta[:, 1],
                             filter_f=filter_f, filter_theta=filter_theta, scale=uq_scale)
        return uq_result

    def significant_blobs(self, sigma_min: float = 1., sigma_max: float = 15., num_sigma: int = 8, k: int = None,
                          ratio: float = 1.):
        alpha = 0.05
        blobs = blob_detection.detect_significant_blobs(alpha=alpha, m=self._m_f, n=self._n_f,
                                                                model=self._linearized_model, x_map=self._x_map_vec,
                                                                sigma_min=sigma_min, sigma_max=sigma_max,
                                                                num_sigma=num_sigma, k=k, ratio=ratio)
        return blobs

    def _uq_dummy(self, options: dict):
        """
        Dummy output for uncertainty quantification. Only for testing purposes. Should be removed before release.
        """
        ci_f = np.column_stack((self.f_map - 1, self.f_map + 1))
        ci_theta = np.column_stack((self.theta_map - 1, self.theta_map + 1))
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        # Reshape
        lower_f = self._reshape_f(ci_f[:, 0])
        upper_f = self._reshape_f(ci_f[:, 1])
        uq_result = UQResult(lower_f=lower_f, upper_f=upper_f, lower_theta=ci_theta[:, 0], upper_theta=ci_theta[:, 1],
                             filter_f=filter_f, filter_theta=filter_theta, scale=1)
        return uq_result

    def _get_filter_function(self, options):
        if self._parameter_map.theta_fixed:
            filter_function, filter_f, filter_vartheta, filter_theta = make_filter_function(m_f=self._m_f,
                                                                                            n_f=self._n_f,
                                                                                            options=options)
        else:
            filter_function, filter_f, filter_vartheta, filter_theta = \
                make_filter_function(m_f=self._m_f, n_f=self._n_f, dim_theta_v=self._parameter_map.dims[1],
                                     options=options)
        return filter_function, filter_f, filter_theta

    def _reshape_f(self, f: np.array):
        """
        Reshapes flattened f into image.
        """
        f_im = np.reshape(f, (self._m_f, self._n_f))
        return f_im

    def make_localization_plot(self, n_sample: int, c_list: Sequence[int], d_list: Sequence[int],
                               scale: float, a: int=1, b: int=1):
        """
        Creates a heuristic localization plot for FCIs based on a random sample of pixels.
        This can be used to tune the localization architecture.

        :param rtol: The relative tolerance for computing credible intervals.
        :param n_sample: Number of pixels used for the random sample.
        :param c_list: List of values for the parameter c, i.e. the vertical radius of the localization window.
        :param d_list: List of values for the parameter d, i.e. the horizontal radius of the localization window.
        :param scale: The scale with respect to which the FCI should be computed.
        :return: computation_times, errors
            - computation_times: A list of the estimated computation time corresponding to each localization architecture.
            - errors: A list of the approximation errors with respect to the baseline FCI (i.e. without any localization).
                The approximation error is defined as the mean Jaccard distance.
        """
        alpha = 0.05
        # Randomly sample pixels.
        all_pixels = np.arange(self._dim_f)
        pixel_sample = np.random.choice(a=all_pixels, size=n_sample, replace=False)
        # Translate pixel_sample to fine scale.
        translated_pixel_sample = self._pixel_to_superpixel(pixel_sample, a, b)
        # For each c-d configuration, compute the FCIs with respect to pixel_sample.
        fci_list = []
        t_list = []
        no_ci = self._dim_f // (a * b)
        for c, d in zip(c_list, d_list):
            options = {"h": scale, "a": a, "b": b, "c": c, "d": d, "sample": translated_pixel_sample}
            # Create appropriate filter
            filter_function, filter_f, filter_theta = self._get_filter_function(options)
            # compute filtered credible intervals
            fci_obj = uq_mode.fci(alpha=alpha, x_map=self._x_map_vec, model=self._linearized_model,
                                  ffunction=filter_function, options=options)
            fci = fci_obj.interval
            t = fci_obj.time_avg
            t_list.append(t * no_ci)   # estimated computation time for complete credible interval
            fci_list.append(fci)

        # Compute baseline.
        options = {"h": scale, "sample": pixel_sample}
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        # compute filtered credible intervals
        fci_base_obj = uq_mode.fci(alpha=alpha, x_map=self._x_map_vec, model=self._linearized_model,
                              ffunction=filter_function, options=options)
        fci_base = fci_base_obj.interval

        # Compute relative localization error
        e_rloc_list = []
        for fci in fci_list:
            mean_jaccard = mean_jaccard_distance(fci, fci_base)
            e_rloc_list.append(mean_jaccard)

        return t_list, e_rloc_list

    def _pixel_to_superpixel(self, pixels, a, b):
        """
        Takes an array of pixels and translated them to corresponding superpixel coordinates.
        :param pixels:
        :return:
        """
        # Translate index to coordinate
        x_coords = pixels % self._n_f
        y_coords = pixels // self._n_f
        # Translate pixel-coordinates to superpixel-coordinates
        y_coords = y_coords // a
        x_coords = x_coords // b
        n_super = self._n_f // b
        # Translate back to indices
        superpixel_indices = n_super * y_coords + x_coords
        return superpixel_indices



