
import numpy as np
import pytest

from uq4pk_fit.uq_mode.evaluation import AffineEvaluationMap, AffineEvaluationFunctional

# Make a dummy affine evaluation functional
class TestFunctional(AffineEvaluationFunctional):
    def __init__(self, dim: int):
        self.dim = dim
        self.phidim = dim
        self.zdim = dim
        self.z0 = np.zeros((dim, ))

    @property
    def u(self):
        return np.identity(self.dim)

    @property
    def v(self):
        return np.zeros((self.dim, ))

    def x(self, z):
        return z

    def phi(self, z: np.ndarray) -> np.ndarray:
        return z

    def lb_z(self, lb: np.ndarray) -> np.ndarray:
        return lb


def test_affine_evaluation_map():
    # create list of affine evaluation functionals
    n = 13
    nofun = 6
    aef_list = []
    for i in range(nofun):
        aef = TestFunctional(dim=n)
        aef_list.append(aef)
    testmap = AffineEvaluationMap(aef_list)
    assert testmap.dim == n
    assert testmap.size == nofun


def test_cannot_initialize_with_empty_list():
    emptylist = []
    with pytest.raises(Exception) as e:
        testmap = AffineEvaluationMap(emptylist)


def test_affine_evaluation_functionals_must_have_same_dim():
    aef1 = TestFunctional(dim=1)
    aef2 = TestFunctional(dim=3)
    with pytest.raises(Exception) as e:
        testmap = AffineEvaluationMap([aef1, aef2])