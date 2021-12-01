
import numpy as np
import os
from typing import List, Type

from ..data import PARAMETER_FILE
from ..trial import Trial
from .test import Test
from .test_result import store_testresult


class SuperTest:

    # ADAPT

    _ChildTest: Type[Test] = None

    def _setup_tests(self) -> List[dict]:
        raise NotImplementedError

    # DO NOT ADAPT:

    def run(self, location: str, trial: Trial):
        """
        Performs all tests for the given trial and stores the results as .pickle-files.
        """
        setup_list = self._setup_tests()
        for data in trial.data_list:
            for setup in setup_list:
                setup = setup | data.setup
                test = self._ChildTest(data=data, setup=setup)
                test_result = test.run()
                store_testresult(savedir=f"{location}/{data.name}/{test.name}", test_result=test_result)
                # also store parameter names
                # also make a parameter names file (needed for summary)
                parameter_names_arr = np.asarray(list(setup.keys()), dtype=str)
                np.savetxt(os.path.join(location, PARAMETER_FILE), parameter_names_arr, delimiter=",", fmt="%s")