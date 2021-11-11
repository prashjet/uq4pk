
from pandas import DataFrame
import os
import numpy as np
import typing

from experiment_kit.test_result_old import TrialResult


class TestSummary:
    def __init__(self, name, parameters: dict, trial_result_list: typing.List[TrialResult]):
        trial_names, avg_trial_values = self._summarize_trials(trial_result_list)
        self._name = name
        names = list(parameters.keys()) + trial_names
        values = list(parameters.values()) + avg_trial_values
        names.insert(0, "name")
        values.insert(0, name)
        self._names = names
        self._values = values

    def get_names(self):
        return self._names.copy()

    def get_values(self):
        return self._values.copy()

    @staticmethod
    def _summarize_trials(trial_result_list):
        summary_names = trial_result_list[0].get_names()
        # make array of all values in trial_result_list
        values = []
        for trial_result in trial_result_list:
            values.append(trial_result.get_values())
        values_arr = np.array(values)
        # compute averages
        value_avg_list = list(np.mean(values_arr, axis=0))
        # compute standard deviations
        value_stdev_list = list(np.std(values_arr, axis=0))
        # add names for the standard deviation
        stdev_names = []
        for name in summary_names:
            stdev_names.append(f"{name}stderr")
        summary_names = summary_names + stdev_names
        summary_values = value_avg_list + value_stdev_list
        assert len(summary_names) == len(summary_values)
        return summary_names, summary_values


class TrialSummary:
    def __init__(self, name, parameters: dict, trial_number, trial_result: TrialResult, ):
        names = list(parameters.keys()) + ["trial"] + trial_result.get_names()
        values = list(parameters.values()) + [trial_number] + trial_result.get_values()
        names.insert(0, "name")
        values.insert(0, name)
        self._names = names
        self._values = values

    def get_names(self):
        return self._names.copy()

    def get_values(self):
        return self._values.copy()



class Experiment:
    """
    ABSTRACT BASE CLASS FOR SUPER-TESTS
    """
    def __init__(self, outname: str, data_list_list: list, name_list: list):
        """
        :param data_list_list: list of lists
            A list with as many entries as snr_list.
            For each element in snr_list, there must be a list of sampled data.
        """
        assert len(name_list) == len(data_list_list)
        self._output_directory = f"{outname}"
        self._data_list_list = data_list_list
        self._name_list = name_list
        self.ChildTest = self._set_child_test()

    # ADAPT:

    def _set_child_test(self):
        raise NotImplementedError

    def _setup_tests(self):
        raise NotImplementedError

    # DO NOT ADAPT:

    def perform_tests(self):
        testsetup_list = self._setup_tests()
        test_summary_list = []
        trial_summary_list = []
        for name, data_list in zip(self._name_list, self._data_list_list):
            print("--------------------")
            print(name)
            print("--------------------")
            test_counter = 1
            for testsetup in testsetup_list:
                setup = testsetup
                testname = testsetup.name
                trial_result_list = []
                trial_counter = 1
                for data in data_list:
                    print("--------------------")
                    print(f"TEST {test_counter} / TRIAL {trial_counter}")
                    print("--------------------")
                    outname = self._create_testname(f"trial{trial_counter}/{name}/{testname}")
                    if not os.path.exists(outname):
                        os.makedirs(outname)
                    trial = self.ChildTest(outname=outname, data=data, setup=setup)
                    trial_result = trial.run()
                    trial_result_list.append(trial_result)
                    # make trial summary
                    trial_summary = TrialSummary(name=name, parameters=testsetup._parameter_list, trial_number=trial_counter,
                                                 trial_result=trial_result)
                    trial_summary_list.append(trial_summary)
                    trial_counter += 1
                # make test summary
                test_summary = TestSummary(name=name, parameters=setup._parameter_list, trial_result_list=trial_result_list)
                test_summary_list.append(test_summary)
                test_counter += 1
        self._create_summary_table(test_summary_list)
        self._create_trials_table(trial_summary_list)

    def _create_testname(self, name):
        return os.path.join(self._output_directory, name)

    def _create_summary_table(self, test_summary_list: typing.List[TestSummary]):
        """
        Creates a nice table lci_vs_fci of the list of TestSummary objects and stores it as csv-file.
        """
        names = test_summary_list[0].get_names()
        dataframe = DataFrame(columns=names)
        i = 0
        for test_summary in test_summary_list:
            dataframe.loc[i] = test_summary.get_values()
            i += 1
        # store the dataframe as csv-file
        dataframe.to_csv(f"{self._output_directory}/summary.csv")

    def _create_trials_table(self, trial_summary_list: typing.List[TrialSummary]):
        """

        :param trial_summary_list: list[TrialSummary]
        :return:
        """
        names = trial_summary_list[0].get_names()
        dataframe = DataFrame(columns=names)
        i = 0
        for test_summary in trial_summary_list:
            dataframe.loc[i] = test_summary.get_values()
            i += 1
        # store the dataframe as csv-file
        dataframe.to_csv(f"{self._output_directory}/trials.csv")