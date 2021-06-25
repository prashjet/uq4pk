"""
Contains classes "TestSetup", "TestResult" and "SuperTest"
"""

import numpy as np
from pandas import DataFrame
import os

from uq4pk_fit.model_fit import *
from uq4pk_fit.model_fit.models.model import FittedModel


class TestSetup:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters


class TestResult:
    def __init__(self):
        self.names = []
        self.attributes = []

    def list(self):
        """
        Returns all attributes as a list
        :return: list
        """
        return self.attributes


class Test:
    def __init__(self, outname, f, setup):
        self.outname = outname
        self.setup = setup
        self.f = f.flatten()
        self.f_im = f
        self.snr = self.setup["snr"]
        expdata = simulate(snr=self.snr, f=self.f_im)
        self._read_ed(ed=expdata)
        self.model = self._initialize_model()

    def do_test(self):
        self._change_model()
        # self._show_model_info()
        fitted_model = self.model.fit()
        # Compute uncertainty quantification
        credible_intervals = self._quantify_uncertainty(fitted_model)
        # Perform some error analysis
        self._error_analysis(fitted_model)
        # Plot everything
        self._plotting(fitted_model, credible_intervals)
        # Make TestResult object and return it
        testresult = self._make_testresult(fitted_model, credible_intervals)
        return testresult

    def _show_model_info(self):
        print("MODEL INFO")
        print(f"delta = {self.model.delta}")
        print(f"alpha1 = {self.model.alpha1}")
        print(f"alpha2 = {self.model.alpha2}")
        print(f"theta_noise = {self.setup['theta_noise']}")
        print(f"theta_v = {self.theta_v}")
        print(f"theta_v_bar = {self.model.theta_v_bar}")
        print(" ")

    def _read_ed(self, ed: ExperimentData):
        self.theta_v = ed.theta_v
        self.y_noi = ed.y_noi
        self.y = ed.y
        self.sdev = ed.sdev
        self.grid = ed.grid

    def _err_f(self, f):
        return np.linalg.norm(f - self.f) / np.linalg.norm(self.f)

    def _err_theta_v(self, theta_v):
        return np.linalg.norm(theta_v - self.theta_v) / np.linalg.norm(self.theta_v)

    def _change_model(self):
        raise NotImplementedError

    def _initialize_model(self):
        raise NotImplementedError

    def _quantify_uncertainty(self, fitted_model: FittedModel):
        raise NotImplementedError

    def _error_analysis(self, fitted_model: FittedModel):
        raise NotImplementedError

    def _plotting(self, fitted_model: FittedModel, credible_intervals):
        raise NotImplementedError

    def _make_testresult(self, fitted_model, credible_intervals) -> TestResult:
        raise NotImplementedError


class SuperTest:

    def __init__(self, output_directory, f_list):
        self.name = output_directory
        self._output_directory = f"out/{output_directory}"
        self.f_list = f_list
        self.ChildTest = Test

    def perform_tests(self):
        testsetup_list = self._setup_tests()
        fno = 0
        testresult_list = []
        for f in self.f_list:
            for testsetup in testsetup_list:
                setup = testsetup.parameters
                testname = testsetup.name
                name = self._create_testname(f"{testname}_f{fno}")
                test = self.ChildTest(outname=name, f=f, setup=setup)
                testresult = test.do_test()
                testresult_list.append(testresult)
            fno += 1
        self._make_table(testresult_list)

    def _setup_tests(self):
        raise NotImplementedError

    def _create_testname(self, name):
        return os.path.join(self._output_directory, name)

    def _make_table(self, testresult_list):
        """
        Creates a nice table out of the list of testresults and stores it as csv-file.
        :param testresult_list: list of TestResult objects
        """
        # fill pandas dataframe with data fromt the testresult_list
        names = testresult_list[0].names
        dataframe = DataFrame(columns=names)
        i = 0
        for testresult in testresult_list:
            dataframe.loc[i] = testresult.list()
            i += 1
        # store the dataframe as csv-file
        dataframe.to_csv(f"{self._output_directory}/{self.name}_results.csv")
