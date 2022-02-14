
import pickle

from .experiment_data import ExperimentData


def load_experiment_data(filename: str) -> ExperimentData:
    """
    Loads an :py:class:`ExperimentData` object from a stored file.

    :param filename: Name of the file.
    :return:
    """
    with open(filename, "rb") as savefile:
        experiment_data = pickle.load(savefile)
    return experiment_data