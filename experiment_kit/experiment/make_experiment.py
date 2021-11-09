
from typing import List

from experiment_kit.experiment.experiment import Experiment
from experiment_kit.test.super_test import SuperTest
from experiment_kit.trial import Trial


def make_experiment(name: str, name_list: List, data_list_list: List[List], supertest: SuperTest) -> Experiment:
    """
    Creates an Experiment object from the given user input.
    """
    # Create all trials.
    listof_trials = []
    for trial_name, data_list in zip(name_list, data_list_list):
        trial = Trial(name=trial_name, data_list=data_list)
        listof_trials.append(trial)
    # Make Experiment object
    experiment = Experiment(name=name, list_of_trials=listof_trials, supertest=supertest)
    return experiment