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
import uq4pk_fit.blob_detection as blob_detection
from .jaccard_distance import mean_jaccard_distance
from .make_filter_function import make_filter_function
from .parameter_map import ParameterMap
from .uq_result import UQResult
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
        Returns MAP estimate for age-metallicity distribution in the correct format!
        """
        f_map_im = np.reshape(self._f_map, (self._m_f, self._n_f))
        return f_map_im

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
            - "kernel": Only relevant for filtered credible intervals. Determines the filter kernel.
                - "gauss": A Gaussian kernel
                - "pixel": A simple delta kernel.
            - "discretization": The employed discretization for the stellar distribution function.
                - "trivial": Just the finest discretization.
                - "window": Discretization with localization windows.
                - "twolevel": Two-levels of discretization.
            - "w1": Only relevant if "discretization" is set to "window" or "twolevel". Determines the vertical
                radius of the localization window.
            - "w2": Same as w1, but for the horizontal radius.
            - "d1": Determines the discretization resolution in the vertical direction.
            - "d2": Determines the discretization resolution in the horizontal direction.
            - "detailed": If yes, the minimizers and maximizers are also returned.
            - "sigma1": Only relevant if "kernel"="gauss". Determines the standard deviation of the Gaussian kernel
                in vertical direction.
            - "sigma2": Same as sigma1, but for the horizontal direction.
        :param Optional[dict] options:
            A dict specifying further options. The possible options depend on the method used (...).
        :returns: UQResult
        """
        if options is None:
            options = {}
        if method == "fci":
            return self._uq_fci(options)
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
        discretization = self._get_discretization(options)
        # Extract the optional weights from the options-dict.
        image_weights = options.setdefault("weights", None)
        if image_weights is not None:
            assert image_weights.shape == (self._m_f, self._n_f)
            weights = image_weights.flatten()
        else:
            weights = None
        # compute filtered credible intervals
        fci_obj = uq_mode.fci(alpha=0.05, x_map=self._x_map_vec, model=self._linearized_model,
                              ffunction=filter_function, discretization=discretization, weights=weights,
                              options=options)
        ci_x = fci_obj.interval
        ci_f, ci_theta = self._parameter_map.ci_f_theta(ci_x)
        uq_scale = options["sigma"]
        # Reshape
        lower_f = self._reshape_f(ci_f[:, 0])
        upper_f = self._reshape_f(ci_f[:, 1])
        uq_result = UQResult(lower_f=lower_f, upper_f=upper_f, lower_theta=ci_theta[:, 0], upper_theta=ci_theta[:, 1],
                             filter_f=filter_f, filter_theta=filter_theta, scale=uq_scale)
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
        # Create appropriate discretization
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

    def rml_sample(self, n_samples: int, sample_prior: bool, options: dict):
        """
        Computes pseudo-samples of the posterior using the randomized maximum-likelihood method.

        :param n_samples: Desired number of samples.
        :param sample_prior: If True, samples the prior guess in every step. If False, only perturbs the measurement.
        :param dict options: Further options for the sampler. See help(uq4pk_fit.uq_mode.rml) for an explanation
            of the possible options.
        :return: Of shape (n, d). The pseudo-samples, where rows correspond to different samples and columns correspond
            to different coordinates.
        """
        samples = uq_mode.rml(model=self._linearized_model, x_map=self._x_map_vec, n_samples=n_samples,
                              sample_prior=sample_prior, options=options)
        return samples

    def fci_from_samples(self, samples: np.ndarray, alpha: float, options: dict = None):
        """
        Estimates a desired FCI from samples.

        :param samples: Of shape (n, model.dim), where n is the number of samples.
        :param alpha: The credibility parameters.
        :param options: Same as for self.uq
        :return: An UQResult object.
        """
        # Initialize filter function.
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        # Get weights.
        image_weights = options.setdefault("weights", None)
        if image_weights is not None:
            assert image_weights.shape == (self._m_f, self._n_f)
            weights = image_weights.flatten()
        else:
            weights = None
        # Estimate FCI using the samples.
        fci_obj = uq_mode.fci_sampling(alpha=alpha, samples=samples, ffunction=filter_function, weights=weights)
        ci_x = fci_obj.interval
        ci_f, ci_theta = self._parameter_map.ci_f_theta(ci_x)
        # Reshape
        lower_f = self._reshape_f(ci_f[:, 0])
        upper_f = self._reshape_f(ci_f[:, 1])
        return lower_f, upper_f


    def compute_fci_stack(self, sigma_list: Sequence[Union[float, np.ndarray]], options: dict = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes a stack of FCIs corresponding to the given list of sigmas.
        WARNING: Only works in the linear case.

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
            Each stack is a sxmxn-array, where each of the s slices corresponds to a lower/upper bound of the
            corresponding FCI.

        """
        if not self._parameter_map.theta_fixed:
            raise NotImplementedError("This method is only implemented for fixed theta.")
        # The FCIs are stored in two lists, one for the lower and one for the upper bound.
        lower_list = []
        upper_list = []
        # For each sigma, compute the corresponding FCI.
        for sigma in sigma_list:
            print(f"Compute FCI for sigma={sigma}.")
            options["sigma"] = sigma
            fci = self._uq_fci(options=options)
            lower_list.append(fci.lower_f)
            upper_list.append(fci.upper_f)
        # Turn lists into corresponding arrays
        lower_stack = np.array(lower_list)
        upper_stack = np.array(upper_list)
        return lower_stack, upper_stack

    def minimize_blobiness(self, alpha: float, sigmas: Sequence):
        """
        Computes the minimally blobby image in the credible region.

        :param alpha: The credibility parameter.
        :param sigmas: The list of the scales.
        :return: f_min. Of shape (m_f, n_f).
        """
        f_min = blob_detection.minimize_blobiness(alpha=alpha, m=self._m_f, n=self._n_f, model=self._linearized_model,
                                                  x_map=self._f_map.flatten(), sigma_list=sigmas)
        return f_min


    def significant_blobs(self, sigmas: Sequence[float], lower_stack: np.ndarray, upper_stack: np.ndarray,
                          reference_image: np.ndarray = None,
                          rthresh: float = 0.01, overlap1: float = 0.1, overlap2: float = 0.5):
        # If no reference image is provided, take MAP as reference:
        if reference_image is None:
            reference_image = self._reshape_f(self._f_map)
        blobs = blob_detection.detect_significant_blobs(sigma_list=sigmas, lower_stack=lower_stack,
                                                        upper_stack=upper_stack, reference=reference_image,
                                                        rthresh=rthresh, overlap1=overlap1, overlap2=overlap2)
        return blobs

    def _uq_dummy(self, options: dict):
        """
        Dummy output for uncertainty quantification. Only for testing purposes. Should be removed before release.
        """
        ci_f = np.column_stack((self._f_map - 1, self._f_map + 1))
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

    def make_localization_plot(self, w1_list: Sequence[int], w2_list: Sequence[int],
                               sigma: Union[float, np.ndarray], discretization_name: str, d1: int=None, d2: int=None,
                               n_sample: int = None):
        """
        Creates a heuristic localization plot for FCIs based on a random sample of pixels.
        This can be used to tune the localization architecture.

        :param n_sample: Number of pixels used for the random sample. If not provided, all pixels will be used in computations.
        :param w1_list: List of values for parameter w1.
        :param w2_list: List of values for parameter w2.
        :param d1: The discretization-resolution in vertical direction.
        :param d2: The discretization-resolution in horizontal direction.
        :return: computation_times, errors
            - computation_times: A list of the estimated computation time corresponding to each localization architecture.
            - errors: A list of the approximation errors with respect to the baseline FCI (i.e. without any localization).
                The approximation error is defined as the mean Jaccard distance.
        """
        alpha = 0.05
        # Randomly sample pixels.
        all_pixels = np.arange(self._dim_f)
        pixel_sample = np.random.choice(a=all_pixels, size=n_sample, replace=False)
        # For each c-d configuration, compute the FCIs with respect to pixel_sample.
        fci_list = []
        t_list = []
        for w1, w2 in zip(w1_list, w2_list):
            options = {"sigma": sigma, "w1": w1, "w2": w2, "d1": d1, "d2": d2, "discretization": discretization_name}
            if n_sample is not None:
                options["sample"] = pixel_sample
            # Create appropriate filter
            filter_function, filter_f, filter_theta = self._get_filter_function(options)
            discretization = self._get_discretization(options)
            # compute filtered credible intervals
            fci_obj = uq_mode.fci(alpha=alpha, x_map=self._x_map_vec, model=self._linearized_model,
                                  ffunction=filter_function, discretization=discretization, options=options)
            fci = fci_obj.interval
            t_list.append(fci_obj.time_avg * self._dim_f)   # estimated computation time for all pixels.
            fci_list.append(fci)

        # ---Compute baseline error.
        options = {"sigma": sigma, "discretization": "trivial"}
        if n_sample is not None: options["sample"] = pixel_sample
        discretization = self._get_discretization(options)
        filter_function, filter_f, filter_theta = self._get_filter_function(options)
        # compute filtered credible intervals
        fci_base_obj = uq_mode.fci(alpha=alpha, x_map=self._x_map_vec, model=self._linearized_model,
                              ffunction=filter_function, discretization=discretization, options=options)
        fci_base = fci_base_obj.interval
        t_base = fci_base_obj.time_avg * self._dim_f

        # Compute relative localization error
        mjd_list = []
        for fci in fci_list:
            mean_jaccard = mean_jaccard_distance(fci, fci_base)
            mjd_list.append(mean_jaccard)
        # Also append baseline
        t_list.append(t_base)
        mjd_list.append(mean_jaccard_distance(fci_base, fci_base))
        return t_list, mjd_list

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



