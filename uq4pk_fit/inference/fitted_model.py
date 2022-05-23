"""
Contains class "FittedModel"
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Sequence, Tuple, Union

import uq4pk_fit.cgn as cgn
from uq4pk_fit.cgn.translator.translator import Translator
import uq4pk_fit.uq_mode as uq_mode
from .jaccard_distance import mean_jaccard_distance
from .make_filter_function import make_filter_function
from .parameter_map import ParameterMap
from .hybrid_discretization import HybridDiscretization


class FittedModel:
    """
    Represents a fitted inference, i.e. a Model linearized around the map:
        0 = mf(x_map) + fwd'(x_map)(x-x_map) + noise,
        noise ~ normal(0, identity / delta^2),
        x ~ normal(x_bar, cov), where cov = (P @ P.T)^(-1),
        x >= lb, A @ x = b.
    """
    def __init__(self, x_map: List[ArrayLike], problem: cgn.Problem, parameter_map: ParameterMap, m_f, n_f, dim_theta,
                 starting_values: List[ArrayLike], scale: float):
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
        self._x_map_vec = self._x_map_vec.clip(self._linearized_model.lb)
        self.scale = scale

    def costfun(self, f, theta_v):
        x = self._parameter_map.x(f, theta_v)
        cost = self._problem.costfun(*x)
        return cost

    @property
    def f_map(self):
        """
        Returns MAP estimate for age-metallicity distribution in the correct format!
        """
        f_map_im = np.reshape(self._f_map, (self._m_f, self._n_f))
        return f_map_im * self.scale

    @property
    def theta_map(self):
        """
        Returns MAP estimate for theta_v
        :return:
        """
        return self._theta_map.copy()

    def pci(self, alpha: float, options: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes pixel-wise credible intervals.

        :param alpha: Crediblity parameter.
        :param options:
             - "discretization": The employed discretization for the stellar distribution function.
                - "trivial": Just the finest discretization.
                - "window": Discretization with localization windows.
                - "twolevel": Two-levels of discretization.
            - "w1": Only relevant if "discretization" is set to "window" or "twolevel". Determines the vertical
                radius of the localization window.
            - "w2": Same as w1, but for the horizontal radius.
            - "d1": Determines the discretization resolution in the vertical direction.
            - "d2": Determines the discretization resolution in the horizontal direction.
            - "weights": Of shape (m, n). If provided, the uncertainty quantification is performed for the rescaled
                image weights * f (pixel-wise multiplication).
        :return: lb, ub
            - lb: Of shape (m, n). Lower bound image of pixel-wise credible interval.
            - ub: Of shape (m, n). Upper bound image of pixel-wise credible interval.
        """
        if options is None:
            options = {}
        filter_function, _, _ = self._get_filter_function({"kernel": "pixel"})
        discretization = self._get_discretization(options)
        fci_obj = self._compute_fci_stack(alpha=alpha, filter_list=[filter_function],
                                          discretization=discretization, options=options)
        lb = self._reshape_stack(stack=fci_obj.lower)
        ub = self._reshape_stack(stack=fci_obj.upper)
        return lb, ub

    def fci(self, alpha: float, sigma: Union[float, np.ndarray], options: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes FCI at a given scale.

        :param alpha: Crediblity parameter.
        :param sigma:
        :param options:
            - "discretization": The employed discretization for the stellar distribution function.
                - "trivial": Just the finest discretization.
                - "window": Discretization with localization windows.
                - "twolevel": Two-levels of discretization.
            - "w1": Only relevant if "discretization" is set to "window" or "twolevel". Determines the vertical
                radius of the localization window.
            - "w2": Same as w1, but for the horizontal radius.
            - "d1": Determines the discretization resolution in the vertical direction.
            - "d2": Determines the discretization resolution in the horizontal direction.
            - "weights": Of shape (m, n). If provided, the uncertainty quantification is performed for the rescaled
                image weights * f (pixel-wise multiplication).
        :return: lb, ub
            - lb: Of shape (m, n). Lower bound image of filtered credible interval.
            - ub: Of shape (m, n). Upper bound image of filtered credible interval.
        """
        lb, ub = self.approx_fci_stack(alpha=alpha, sigma_list=[sigma], options=options)
        return lb, ub

    def approx_fci_stack(self, alpha: float, sigma_list: Sequence[Union[float, np.ndarray]], options: dict = None):
        """
        Computes FCI stack using special discretization.
        WARNING: Only works in the linear case.

        :param alpha: Crediblity parameter.
        :param sigma_list: The list of sigma-values for which a FCI has to be computed.
        :param options:
            - "discretization": The employed discretization for the stellar distribution function.
                - "trivial": Just the finest discretization.
                - "window": Discretization with localization windows.
                - "twolevel": Two-levels of discretization.
            - "w1": Only relevant if "discretization" is set to "window" or "twolevel". Determines the vertical
                radius of the localization window.
            - "w2": Same as w1, but for the horizontal radius.
            - "d1": Determines the discretization resolution in the vertical direction.
            - "d2": Determines the discretization resolution in the horizontal direction.
            - "weights": Of shape (m, n). If provided, the uncertainty quantification is performed for the rescaled
                image weights * f (pixel-wise multiplication).
        :return: lower_stack, upper_stack
            - lower_stack: Of shape (k, m, n).
            - upper_stack: Of shape (k, m, n).
        """
        if not self._parameter_map.theta_fixed:
            raise NotImplementedError("This method is only implemented for fixed theta.")
        # Create filter functions.
        filter_list = []
        for sigma in sigma_list:
            options["sigma"] = sigma
            filter_function, filter_f, filter_theta = self._get_filter_function(options)
            filter_list.append(filter_f)
        # Create discretization.
        discretization = self._get_discretization(options)
        # Compute FCI-stack.
        fci_obj = self._compute_fci_stack(alpha=alpha, filter_list=filter_list,
                                          discretization=discretization, options=options)
        lower_stack = self._reshape_stack(stack=fci_obj.lower)
        upper_stack = self._reshape_stack(stack=fci_obj.upper)
        return lower_stack, upper_stack

    def fci_stack(self, alpha: float, sigma_list: Sequence[Union[float, np.ndarray]], options: dict = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes a stack of FCIs using some additional speedups that assuming a trivial discretization.

        :param alpha: Crediblity parameter.
        :param sigma_list: List of sigma parameters for each filter.
        :return: lower_stack, upper_stack
            Each stack is a sxmxn-array, where each of the s slices corresponds to a lower/upper bound of the
            corresponding FCI.

        """
        if options is None:
            options = {}
        if not self._parameter_map.theta_fixed:
            raise NotImplementedError("This method is only implemented for fixed theta.")
        # Make list of filtered functions.
        filter_list = []
        for sigma in sigma_list:
            opt = {"sigma": sigma}
            filter_function, filter_f, filter_theta = self._get_filter_function(opt)
            filter_list.append(filter_f)

        fci_obj = uq_mode.fci_stack(alpha=alpha, model=self._linearized_model, x_map=self._x_map_vec,
                                    ffunction_list=filter_list, options=options)
        lower_stack = self._reshape_stack(stack=fci_obj.lower)
        upper_stack = self._reshape_stack(stack=fci_obj.upper)
        return lower_stack, upper_stack

    def make_localization_plot(self, alpha: float, w1_list: Sequence[int], w2_list: Sequence[int],
                               sigma: Union[float, np.ndarray], discretization_name: str, d1: int = None,
                               d2: int = None, n_sample: int = None):
        """
        Creates a heuristic localization plot for FCIs based on a random sample of pixels.
        This can be used to tune the localization architecture.

        :param alpha: Crediblity parameter.
        :param sigma: sigma-parameter for Gaussian filter.
        :param n_sample: Number of pixels used for the random sample. If not provided, all pixels will be used in
            computations.
        :param w1_list: List of values for parameter w1.
        :param w2_list: List of values for parameter w2.
        :param d1: The discretization-resolution in vertical direction.
        :param d2: The discretization-resolution in horizontal direction.
        :param discretization_name: Name of the discretization you want to use, i.e. "trivial", "window", or "twolevel".
        :return: computation_times, errors
            - computation_times: A list of the estimated computation time corresponding to each localization
                architecture.
            - errors: A list of the approximation errors with respect to the baseline FCI (i.e. without any
                localization). The approximation error is defined as the mean Jaccard distance.
        """
        # Randomly sample pixels.
        all_pixels = np.arange(self._dim_f)
        pixel_sample = np.random.choice(a=all_pixels, size=n_sample, replace=False)

        # --- For each c-d configuration, compute the FCIs with respect to pixel_sample.
        fci_list = []
        t_list = []
        for w1, w2 in zip(w1_list, w2_list):
            options = {"sigma": sigma, "w1": w1, "w2": w2, "d1": d1, "d2": d2, "discretization": discretization_name,
                       "use_ray": True}
            if n_sample is not None:
                options["sample"] = pixel_sample
            # Create appropriate filter
            filter_function, filter_f, filter_theta = self._get_filter_function(options)
            discretization = self._get_discretization(options)
            # compute filtered credible intervals
            fci_obj = self._compute_fci_stack(alpha=alpha, filter_list=[filter_function],
                                              discretization=discretization, options=options)
            lb = fci_obj.lower
            ub = fci_obj.upper
            t_avg = fci_obj.time_avg
            t_list.append(t_avg * self._dim_f)   # estimated computation time for all pixels.
            fci_list.append(np.column_stack([lb, ub]))

        # --- Compute baseline.
        options = {"sigma": sigma, "discretization": "trivial", "use_ray": True}
        if n_sample is not None:
            options["sample"] = pixel_sample
        discretization = self._get_discretization(options)
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        fci_obj = self._compute_fci_stack(alpha=alpha, filter_list=[filter_function],
                                          discretization=discretization, options=options)
        lb = fci_obj.lower
        ub = fci_obj.upper
        fci_base = np.column_stack([lb, ub])
        t_base = fci_obj.time_avg * self._dim_f

        # --- Compute relative localization error
        mjd_list = []
        for fci in fci_list:
            mean_jaccard = mean_jaccard_distance(fci, fci_base)
            mjd_list.append(mean_jaccard)
        # Also append baseline
        t_list.append(t_base)
        mjd_list.append(mean_jaccard_distance(fci_base, fci_base))
        return t_list, mjd_list

    def marginal_credible_intervals(self, alpha: float, axis: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes marginal (1-alpha)-credible interval for the image f along given axis.

        :param alpha: Crediblity parameter.
        :param axis: The axis over which we sum. Since f is two-dimensional, this should be either 0 or 1.
        :return: lb, ub
            - lb: Vector that gives the lower bounds of the credible intervals.
            - ub: Vector that gives the upper bounds of the credible intervals.
        """
        # Check input.
        assert 0 < alpha < 1.
        assert axis in [0, 1]
        # Set up the marginalization filter.
        marginalization_ffunction = uq_mode.MarginalizingFilterFunction(shape=(self._m_f, self._n_f), axis=axis)
        # Compute using uq_mode.fci.
        options = {"optimizer": "SCS", "use_ray": True}
        fci_obj = uq_mode.fci(alpha=alpha, model=self._linearized_model, x_map=self._x_map_vec,
                              filter_function=marginalization_ffunction, options=options)
        # Postprocess the output and return.
        lb = self.scale * fci_obj.lower.flatten()
        ub = self.scale * fci_obj.upper.flatten()
        return lb, ub

    # PROTECTED

    def _compute_fci_stack(self, alpha: float, filter_list, discretization, options) -> uq_mode.FCI:
        """

        :param alpha: Crediblity parameter.
        :param filter_list:
        :param discretization:
        :param options:
        :return:
        """
        # Create downsampling object.
        downsampling = self._setup_downsampling(options)
        # Compute FCIs (creates an object of type `uq_mode.FCI`).
        fci_obj = uq_mode.adaptive_fci(alpha=alpha, model=self._linearized_model, x_map=self._x_map_vec,
                                       filter_functions=filter_list, discretization=discretization,
                                       downsampling=downsampling, options=options)
        # Rescale
        fci_obj.lower *= self.scale
        fci_obj.upper *= self.scale
        return fci_obj

    def _setup_downsampling(self, options) -> uq_mode.Downsampling:
        """
        Reads the options and sets up downsampling, if needed.

        :param options:
        :return:
        """
        a = options.setdefault("a", None)
        b = options.setdefault("b", None)
        if a is not None and b is not None:
            downsampling = uq_mode.RectangularDownsampling(shape=(self._m_f, self._n_f), a=a, b=b)
        else:
            downsampling = None
        return downsampling

    def _reshape_stack(self, stack: np.ndarray):
        """

        :param stack: Of shape (k, d).
        :return: Of shape (k, m, n).
        """
        assert stack.ndim in [1, 2]
        if stack.ndim == 1:
            # If stack is one-dimensional, reshape into single image.
            image_stack = stack.reshape(self._m_f, self._n_f)
        else:
            # Have to reshape into stack of images.
            num_scales = stack.shape[0]
            image_stack = stack.reshape(num_scales, self._m_f, self._n_f)
        return image_stack

    def _get_discretization(self, options: dict) -> uq_mode.AdaptiveImageDiscretization:
        discretization_name = options.setdefault("discretization", "trivial")
        d1 = options.setdefault("d1", 1)
        d2 = options.setdefault("d2", 1)
        w1 = options.setdefault("w1", 1)
        w2 = options.setdefault("w2", 1)
        if discretization_name == "trivial":
            f_discretization = uq_mode.TrivialAdaptiveDiscretization(dim=self._dim_f)
        elif discretization_name == "window":
            f_discretization = uq_mode.LocalizationWindows(im_ref=self._reshape_f(self._f_map), w1=w1, w2=w2)
        elif discretization_name == "twolevel":
            f_discretization = uq_mode.AdaptiveTwoLevelDiscretization(im_ref=self._reshape_f(self._f_map), d1=d1, d2=d2,
                                                                      w1=w1, w2=w2)
        else:
            raise KeyError("Unknown value for parameter 'discretization'.")
        # If theta is not fixed, we have to use a hybrid discretization.
        if self._parameter_map.theta_fixed:
            discretization = f_discretization
        else:
            discretization = HybridDiscretization(f_discretization=f_discretization,
                                                  dim_theta=self._parameter_map.dims[1])
        return discretization

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
