
import numpy as np
from typing import List

from .affine_evaluation_functional import AffineEvaluationFunctional


class AffineEvaluationMap:
    """
    An affine evaluation map is simply a collection of affine evaluation functionals plus a way to map the phi-values
    to a parameter vector x.
    """
    dim: int                                       # The dimension of the parameter space (where x lives).
    phidim: int                                    # The sum of the phi_values for all affine evaluation functionals.
    aef_list: List[AffineEvaluationFunctional]     # The associated list of evaluation functionals.

    def __init__(self, aef_list: List[AffineEvaluationFunctional]):
        # Check the evaluation functionals for consistency
        self._check_input(aef_list)
        self.aef_list = aef_list
        self.dim = aef_list[0].dim
        # Compute phidim
        self.phidim = 0
        for aef in aef_list:
            self.phidim += aef.phidim

    @property
    def size(self) -> int:
        """
        Returns the number of affine evaluation functionals in the map.
        """
        return len(self.aef_list)

    def select(self, indices: np.array):
        """
        Kicks out all the affine evaluation functionals that do not correspond to an index in 'indices'.

        :param indices: List ouf indices that should be kept.
        """
        new_aef_list = [self.aef_list[i] for i in indices]
        # Have to recompute phidim.
        new_phdim = 0
        for aef in new_aef_list:
            new_phdim += aef.phidim
        self.aef_list = new_aef_list
        self.phidim = new_phdim

    def _check_input(self, aef_list: List[AffineEvaluationFunctional]):
        # list must not be empty
        if len(aef_list) == 0:
            raise Exception("'aef_list' must not be empty.")
        # all evaluation functionals must have same dim parameter
        dim = aef_list[0].dim
        for aef in aef_list:
            if aef.dim != dim:
                raise Exception("All functionals in 'aef_list' must have same dimension.")


