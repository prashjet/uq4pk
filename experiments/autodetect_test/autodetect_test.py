
from math import log
import numpy as np

from uq4pk_fit.inference import *
from experiment_kit import *


class AutodetectTest(Test):

    def _read_setup(self, setup: dict):
        self.h = setup["h"]

    def _create_name(self) -> str:
        name = str(self.h)
        return name

    def _change_model(self):
        self.model.normalize()
        # Make linear
        self.model.fix_theta_v(indices=np.arange(self.dim_theta), values=self.theta_true)

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        s = max(np.ceil(self.h).astype(int), 3)
        uq = fitted_model._uq_fci(options={"kernel": "gauss", "a": 1, "b": 1, "c": s, "d": 2 * s, "h": self.h})
        return uq


class AutodetectSupertest(SuperTest):

    _ChildTest = AutodetectTest

    def _setup_tests(self):
        setup_list = []
        maxscale = 10.
        minscale = 1.
        K = 1.6
        nscales = np.floor((log(maxscale) - log(minscale)) / log(K)).astype(int)
        scales = [minscale * K ** i for i in range(nscales + 1)]
        for h in scales:
            setup_list.append({"h": h})
        return setup_list