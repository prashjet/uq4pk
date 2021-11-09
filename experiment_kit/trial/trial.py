
from typing import List

from experiment_kit.experiment.experiment_data import ExperimentData


class Trial:
    """
    Trial represents a way to structure tests that are not comparable, e.g. with respect to SNR etc.
    """
    def __init__(self, name: str, data_list: List[ExperimentData]):
        self.name = str(name)
        self.data_list = data_list