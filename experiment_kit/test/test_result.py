
import numpy as np
import os
import pickle

from uq4pk_fit.inference import UQResult, FittedModel, StatModel
from experiment_kit.experiment.experiment_data import ExperimentData

from ..data import TESTRESULT_FILE


class TestResult:

    def __init__(self, data: ExperimentData, parameters: dict, statmodel: StatModel, fitted_model: FittedModel,
                 uq: UQResult):
        self.data = data
        self.parameters = parameters
        self.f_map = fitted_model.f_map
        self.theta_map = fitted_model.theta_map
        self.cost_map = fitted_model.costfun(self.f_map, self.theta_map)
        self.cost_true = fitted_model.costfun(self.data.f_true, self.data.theta_true)
        self.cost_ref = fitted_model.costfun(self.data.f_ref, self.data.theta_guess)
        if uq.lower_f is None:
            self.ci_f = None
        else:
            self.ci_f = np.column_stack([uq.lower_f, uq.upper_f])
        if uq.lower_theta is None:
            self.ci_theta = None
        else:
            self.ci_theta = np.column_stack([uq.lower_theta, uq.upper_theta])
        self.features = uq.features
        filter_f = uq.filter_f
        self.uq_scale = uq.scale
        if filter_f is not None:
            self.phi_map = filter_f.evaluate(self.f_map)
            self.phi_true = filter_f.evaluate(self.data.f_true)
            self.phi_ref = filter_f.evaluate(self.data.f_ref)
        else:
            self.phi_map = self.f_map
            self.phi_true = self.data.f_true
            self.phi_ref = self.data.f_ref
        assert self.phi_map.size == self.f_map.size
        assert self.phi_true.size == self.f_map.size
        assert self.phi_ref.size == self.f_map.size
        self.regop_theta = statmodel.P2.mat
        self.m_f = statmodel.m_f
        self.n_f = statmodel.n_f
        self.rdm_map = self._rdm(self.f_map, self.theta_map, fitted_model, statmodel)
        self.rdm_true = self._rdm(self.data.f_true, self.data.theta_true, fitted_model, statmodel)
        self.rdm_ref = self._rdm(self.data.f_ref, self.data.theta_guess, fitted_model, statmodel)
        # Ticks needed for correct visualization
        self.ssps = data.forward_operator.modgrid

    def image(self, f):
        m_f = self.m_f
        n_f = self.n_f
        f_im = np.reshape(f, (m_f, n_f))
        return f_im

    def _rdm(self, f, theta, fitted_model: FittedModel, statmodel: StatModel):
        x = statmodel._parameter_map.x(f, theta)
        misfit = fitted_model._problem.q.fwd(fitted_model._problem.fun(*x))
        rdm = np.linalg.norm(misfit) / np.sqrt(fitted_model._problem.scale)
        return rdm




def store_testresult(savedir: str, test_result: TestResult):
    """
    Stores a TestResult object in a specified folder using pickle.
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(f"{savedir}/{TESTRESULT_FILE}", "wb") as savefile:
        pickle.dump(test_result, savefile)
        print(f"Stored in '{savedir}'.")


def load_testresult(savedir) -> TestResult:
    """
    Loads a TestResult object from the given file.
    """
    with open(f"{savedir}/{TESTRESULT_FILE}", "rb") as savefile:
        test_result = pickle.load(savefile)
    return test_result

