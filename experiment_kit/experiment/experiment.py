
import glob
import numpy as np
import os
import pandas
from typing import List

from ..data import EVALUATION_FILE, PARAMETER_FILE, SUMMARY_FILE, TESTRESULT_FILE
from ..test import SuperTest, load_testresult
from ..evaluation import evaluate_testresult
from ..plotting import plot_testresult
from ..trial import Trial


class Experiment:

    def __init__(self, name: str, list_of_trials: List[Trial], supertest: SuperTest):
        self._name = name
        self._trials = list_of_trials
        self._supertest = supertest
        self._run_successful = False

    def clean(self):
        # removes all pickle files, because they take so much space
        list_of_savedirs = self._find_savedirs(TESTRESULT_FILE)
        for savedir in list_of_savedirs:
            filename = os.path.join(savedir, TESTRESULT_FILE)
            print(f"Removing {filename}.")
            os.remove(filename)

    def do_all(self):
        """
        Summary command. Executes ``run``, ``evaluate``, ``plot``, and ``make_summary``. But does not clean.
        """
        self.run()
        self.evaluate()
        self.plot()
        self.make_summary()

    def run(self):
        """
        Runs all trials and writes output in "name/out_old".
        Only produces the minimal data, before evaluation, i.e. all the things that test_result depends on.
        """
        for trial in self._trials:
            print(f"Run supertest on trial {trial.name}")
            self._supertest.run(location=f"out/{self._name}/{trial.name}", trial=trial)

    def evaluate(self):
        list_of_savedirs = self._find_savedirs(TESTRESULT_FILE)
        for savedir in list_of_savedirs:
            print("Evaluating test result in " + savedir)
            test_result = load_testresult(savedir)
            evaluate_testresult(savedir=savedir, test_result=test_result)

    def make_summary(self):
        """
        After experiment.evaluate has been executed, makes a file "summary.csv" and stores it in "{self.name}/out".
        """
        # Iterate over trials
        for trial in self._trials:
            # Get all directories that contain an evaluation file
            list_of_savedirs = self._find_savedirs(EVALUATION_FILE, loc=trial.name)
            # Read all result-dataframes and combine them.
            list_of_results = []
            for savedir in list_of_savedirs:
                result = pandas.read_csv(os.path.join(savedir, EVALUATION_FILE), index_col=[0])
                list_of_results.append(result)
            combined_result = pandas.concat(list_of_results)
            # Next, average results (separately for each parameter value)
            # For this, we need to know the exact parameter names.
            parameter_names = np.loadtxt(f"out/{self._name}/{trial.name}/{PARAMETER_FILE}", dtype=str).tolist()
            summary = self._summarize(combined_result, parameter_names)
            summary.to_csv(f"out/{self._name}/{trial.name}/{SUMMARY_FILE}")

    def plot(self, extra_scale: float = None):
        list_of_savedirs = self._find_savedirs(TESTRESULT_FILE)
        for savedir in list_of_savedirs:
            test_result = load_testresult(savedir)
            plot_testresult(savedir=savedir, test_result=test_result, extra_scale=extra_scale)

    def _summarize(self, result: pandas.DataFrame, parameter_names: list):
        """
        Averages a given result with respect to the given parameters. That is, only rows with matching parameter
        values are summed over. Also computes the standard error.
        """
        averaged_result = result.groupby(parameter_names).mean()
        stderr_result = result.groupby(parameter_names).std()
        # Concatenate standard errors to the average-dataframe, by appending "_std" to each column name.
        column_names = list(averaged_result.columns)
        stderr_column_names = []
        for name in column_names:
            stderr_name = name + "std"
            stderr_column_names.append(stderr_name)
        assert len(column_names) == len(stderr_column_names)
        stderr_result.columns = stderr_column_names
        summary = averaged_result.join(stderr_result)
        return summary

    def _find_savedirs(self, filename: str, loc: str = None) -> List[str]:
        """
        Returns a list of all directories in the path "out/{self._name}" that contain a file with name ``filename``.
        """
        if loc is None:
            location = "/"
        else:
            location = "/" + loc + "/"
        list_of_savedirs = []
        for fullpath in glob.glob("out" + "/" + self._name + location + "**" + "/" + filename, recursive=True):
            savedir = fullpath.removesuffix("/" + filename)
            list_of_savedirs.append(savedir)
        return list_of_savedirs
