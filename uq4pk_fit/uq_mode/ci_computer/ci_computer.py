
import numpy as np
import ray
from typing import Sequence

from ..linear_model import CredibleRegion, LinearModel
from ..evaluation import AffineEvaluationMap
from ..optimization import socp_solve, socp_solve_remote, ECOS, SCS
from .progress_bar import ProgressBar
from .create_socp import create_socp


RTOL = 0.001     # relative tolerance for the cost-constraint
RACC = 1e-2      # relative accuracy for optimization solvers.
NUM_CPUS = 7


class CIComputer:
    """
    Manages the computation of credible intervals from evaluation maps.

    Given a linear model and an evaluation map, the CI computer solves for each evaluation object (loss, x, phi)
    the optimization problems
    min_z / max_z w^\top z s.t. AU z = b - A v, ||C z - d||_2^2 <= e.
    and returns the credible interval [phi(z_min), phi(z_max)].
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, aemap_list: Sequence[AffineEvaluationMap],
                 options: dict):
        # Create list of affine evaluation maps from discretization.
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._num_scales = len(aemap_list)
        self._num_pixels = len(aemap_list[0].aef_list)
        self._aemap_list = aemap_list
        cost_map = model.cost(x_map)
        self._ctol = RTOL * cost_map

        self._cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)
        # Read options.
        if options is None:
            options = {}
        self._use_ray = options.setdefault("use_ray", True)
        self._num_cpus = options.setdefault("num_cpus", NUM_CPUS)
        optimizer_name = options.setdefault("optimizer", "ECOS")
        self._eps = options.setdefault("eps", 1e-4)
        if optimizer_name == "ECOS":
            print("Using ECOS.")
            self._Optimizer = ECOS
        elif optimizer_name == "SCS":
            print("Using SCS.")
            self._Optimizer = SCS
        else:
            raise ValueError("Unknown optimizer.")
        self._cino = self._num_scales * self._num_pixels  # number of credible intervals
        self._optno = 2 * self._cino                      # number of optimization problems to solve
        # Setting up progress bar
        if self._use_ray:
            ray.shutdown()
            ray.init(num_cpus=self._num_cpus)
            self._pb = ProgressBar(self._optno)
            self._actor = self._pb.actor

    def compute_all(self):
        """
        Computes all credible intervals.

        :returns: Object of type :py:class:`CredibleInterval`
        """
        # Initialize containers for results.
        if self._use_ray:
            out_lower_list, out_upper_list = self._compute_with_ray()
        else:
            out_lower_list, out_upper_list = self._compute_without_ray()
        # Read out results.
        lower_list = [out.values for out in out_lower_list]  # This is now a list of lists.
        upper_list = [out.values for out in out_upper_list]
        times_for_lower = [out.time for out in out_lower_list]
        times_for_upper = [out.time for out in out_upper_list]
        times = [t1 + t2 for t1, t2 in zip(times_for_lower, times_for_upper)]
        # The results are now converted to arrays.
        # Since out_lower_list_result is a list of length self._num_pixels, with each element another list of length
        # self._num_scales, the resulting array will be of the shape (self._num_pixels, self._num_scales).
        # We want it the other way around, so we transpose.
        lower_stack = np.array(lower_list).T
        # Same for upper stack.
        upper_stack = np.array(upper_list).T
        # We also compute the average time per FCI.
        time_avg = np.mean(np.array(times))
        assert np.all(lower_stack <= upper_stack + 1e-8)
        # Enforce ub >= lb for each ub, lb in upper_stack, lower_stack (because even if there is only a small error,
        # it can mess up the subsequent computations.
        upper_stack = upper_stack.clip(min=lower_stack)

        return lower_stack, upper_stack, time_avg

    def _compute_with_ray(self):
        print("Starting computations...")
        out_lower_ids = []
        out_upper_ids = []
        for i in range(self._num_pixels):
            # Create list of all affine evaluation functionals at that pixels.
            aef_list_i = [aemap.aef_list[i] for aemap in self._aemap_list]
            aef_0i = aef_list_i[0]
            # Create SOCP for i-th pixel.
            socp_i = create_socp(aefun=aef_0i, credible_region=self._cregion)
            # Now compute list of lower and upper bounds (each will have length self._num_cpus).
            # Note that the computation time is measured aswell.
            # This is relevant for heuristic tuning of localization, etc.
            socp_i1 = ray.put(socp_i)
            socp_i2 = ray.put(socp_i)
            aef_list_i_id1 = ray.put(aef_list_i)
            aef_list_i_id2 = ray.put(aef_list_i)
            opt1 = ray.put(self._Optimizer(eps=self._eps))
            opt2 = ray.put(self._Optimizer(eps=self._eps))
            out_lower = socp_solve_remote.remote(aef_list=aef_list_i_id1, socp=socp_i1, optimizer=opt1, ctol=self._ctol,
                                          actor=self._actor, mode="min")
            out_upper = socp_solve_remote.remote(aef_list=aef_list_i_id2, socp=socp_i2, optimizer=opt2, ctol=self._ctol,
                                          actor=self._actor, mode="max")
            # Store the results in the containers.
            out_lower_ids.append(out_lower)
            out_upper_ids.append(out_upper)
        # Now, we "get" the results.
        # If we use Ray, the actual computations start here.
        print("Starting computations...")
        self._pb.print_until_done()
        out_lower_list = ray.get(out_lower_ids)
        out_upper_list = ray.get(out_upper_ids)
        ray.shutdown()
        return out_lower_list, out_upper_list

    def _compute_without_ray(self):
        out_lower_list = []
        out_upper_list = []
        optimizer = self._Optimizer()
        for i in range(self._num_pixels):
            # Create list of all affine evaluation functionals at that pixels.
            aef_list_i = [aemap.aef_list[i] for aemap in self._aemap_list]
            aef_0i = aef_list_i[0]
            # Create SOCP for i-th pixel.
            socp_i = create_socp(aefun=aef_0i, credible_region=self._cregion)
            # Now compute list of lower and upper bounds (each will have length self._num_cpus).
            # Note that the computation time is measured aswell.
            # This is relevant for heuristic tuning of localization, etc.
            out_lower = socp_solve(aef_list=aef_list_i, socp=socp_i, mode="min", optimizer=optimizer, ctol=self._ctol)
            out_upper = socp_solve(aef_list=aef_list_i, socp=socp_i, mode="max", optimizer=optimizer, ctol=self._ctol)
            self._print_status(i, t=out_lower.time + out_upper.time)
            # Store the results in the containers.
            out_lower_list.append(out_lower)
            out_upper_list.append(out_upper)
        return out_lower_list, out_upper_list

    def _print_status(self, i: int, t: float):
        """
        Displays the current status of the computation.
        """
        print("\r", end="")
        print(f"Computing credible interval for pixel {i+1}/{self._dim}. Average time: {t:.3f} s.", end=" ")
