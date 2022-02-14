
import pickle

from .experiment_data import ExperimentData


def save_experiment_data(experiment_data: ExperimentData, savename: str):
    """
    Stores experiment data as a pickle-file.

    :param experiment_data: The experiment data.
    :param savename: The relative path to the desired save location.
    """
    with open(savename, "wb") as savefile:
        pickle.dump(experiment_data, savefile)