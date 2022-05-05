"""
Variant of CI computer that computes the filtered credible interval using warm-starts.
"""


import numpy as np
from typing import List
from time import time
import ray

from ..linear_model import CredibleRegion, LinearModel
from ..evaluation import AffineEvaluationMap
from ..optimization import SCS, ECOS, solve_distributed_remote, solve_distributed
from .progress_bar import ProgressBar
from .create_socp import create_socp


NUM_CPU = 7


RTOL = 0.01     # relative tolerance for the cost-constraint
RACC = 1e-2      # relative accuracy for optimization solvers.


class StackComputer:
    """
    Manages the computation of credible intervals from evaluation maps.

    Given a linear model and an evaluation map, the CI computer solves for each evaluation object (loss, x, phi)
    the optimization problems
    min_z / max_z w^\top z s.t. AU z = b - A v, ||C z - d||_2^2 <= e.
    and returns the credible interval [phi(z_min), phi(z_max)].
    """
    def __init__(self, alpha: float, model: LinearModel, x_map: np.ndarray, aemap_list: List[AffineEvaluationMap],
                 scale: float, options: dict):
        # precompute:
        self._alpha = alpha
        self._x_map = x_map.copy()
        self._dim = x_map.size
        self._aemap_list = aemap_list
        self._num_scales = len(aemap_list)
        cost_map = model.cost(x_map)
        self._ctol = RTOL * cost_map
        self._abstol = RACC * scale
        self._cregion = CredibleRegion(alpha=alpha, model=model, x_map=x_map)
        cost_map = model.cost(x_map)
        self._ctol = RTOL * cost_map
        # Initialize ray
        self._num_cpus = NUM_CPU
        self._use_ray = options.setdefault("use_ray", True)
        optimizer = options.setdefault("optimizer", "SCS")
        if optimizer == "SCS":
            self._Optimizer = SCS
        elif optimizer == "ECOS":
            self._Optimizer = ECOS
        else:
            raise ValueError("Unknown value for key 'optimizer'.")
        self._Optimizer = SCS
        self._optno = 2 * self._num_scales * self._dim
        if self._use_ray:
            ray.shutdown()
            ray.init(num_cpus=NUM_CPU)
            self._pb = ProgressBar(self._optno)
            self._actor = self._pb.actor

    def compute_all(self):
        """
        Computes all credible intervals.

        :returns: Object of type :py:class:`CredibleInterval`
        """
        # Create first SOCP.
        aefun_0 = self._aemap_list[0].aef_list[0]
        t0 = time()
        socp = create_socp(aefun_0, self._cregion)
        t1 = time()
        print(f"Time for initializing SOCP: {t1 - t0}.")
        # Partition problems.
        pixels_per_cpu = np.ceil(self._dim / self._num_cpus).astype(int)
        n_part = [j * pixels_per_cpu for j in range(self._num_cpus)]
        n_part.append(self._dim)    # Last pixel
        # Create aef list for each individual CPU.
        aef_list_for_cpu = []
        for k in range(self._num_cpus):
            index_set = range(n_part[k], n_part[k+1])
            scale_set = range(self._num_scales) # all scales
            aef_list_k = [[self._aemap_list[j].aef_list[i] for j in scale_set] for i in index_set]
            aef_list_for_cpu.append(aef_list_k)
        lower_result_list = []
        upper_result_list = []
        for k in range(self._num_cpus):
            if self._use_ray:
                socp_id = ray.put(socp)
                opt1 = ray.put(self._Optimizer())
                opt2 = ray.put(self._Optimizer())
                lower_result = solve_distributed_remote.remote(socp=socp_id, aef_list_list=aef_list_for_cpu[k],
                                                               optimizer=opt1, ctol=self._ctol, actor=self._actor,
                                                               mode="min")
                upper_result = solve_distributed_remote.remote(socp=socp_id, aef_list_list=aef_list_for_cpu[k],
                                                               optimizer=opt2, ctol=self._ctol, actor=self._actor,
                                                               mode="max")
                lower_result_list.append(lower_result)
                upper_result_list.append(upper_result)
            else:
                opt = self._Optimizer()
                n_pixels = len(aef_list_for_cpu[k])
                n_scales = len(aef_list_for_cpu[k][0])
                lower_result = solve_distributed(socp=socp, aef_list_list=aef_list_for_cpu[k], optimizer=opt,
                                                 ctol=self._ctol, mode="min")
                upper_result = solve_distributed(socp=socp, aef_list_list=aef_list_for_cpu[k], optimizer=opt,
                                                 ctol=self._ctol,  mode="max")
                assert lower_result.shape == (n_scales, n_pixels)
                assert lower_result.shape == (n_scales, n_pixels)
                lower_result_list.append(lower_result)
                upper_result_list.append(upper_result)
        t0 = time()
        if self._use_ray:
            self._pb.print_until_done()
            lower_list = ray.get([lower_result for lower_result in lower_result_list])
            upper_list = ray.get([upper_result for upper_result in upper_result_list])
        else:
            lower_list = [lower_result for lower_result in lower_result_list]
            upper_list = [upper_result for upper_result in upper_result_list]
        t1 = time()
        print(f"Done. Total time = {t1 - t0}s.")

        # Assemble solution.
        lower_stack = np.column_stack(lower_list)
        upper_stack = np.column_stack(upper_list)

        assert np.all(lower_stack <= upper_stack + 1e-5)

        return lower_stack, upper_stack