
import numpy as np
import pytest

from uq4pk_fit.cgn import DiagonalOperator
from uq4pk_fit.uq_mode.linear_model import LinearModel


@pytest.fixture
def dummy_model():
    n = 3
    m = 2
    sigma = 0.1
    h = np.random.randn(m, n)
    x = np.array([1, 2])
    lb = np.array([-1, -1])
    y = h @ x + sigma * np.random.randn(m, n)
    q = DiagonalOperator(dim=n, s=1 / sigma)
    pprec = np.array([1., 2.])
    r = DiagonalOperator(dim=n, s=pprec)
    lin_model = LinearModel(h=h, y=y, q=q, m=np.zeros((n,)), r=r, a=None, b=None, lb=None)
    return lin_model


def test_linear_model_cost_and_cost_grad():
    n = 3
    m = 2
    sigma = 0.1
    h = np.random.randn(m, n)
    x = np.array([1, 2, 2])
    lb = np.array([-1, -1, -1])
    y = h @ x + sigma * np.random.randn(m)
    q = DiagonalOperator(dim=n, s=1 / sigma)
    pprec = np.array([1., 2., 2.])
    r = DiagonalOperator(dim=n, s=pprec)
    lin_model = LinearModel(h=h, y=y, q=q, m=np.zeros((n,)), r=r, a=None, b=None, lb=lb)
    def costfun(z):
        return 0.5 * np.sum(np.square((h @ z - y) / sigma)) + 0.5 * np.sum(np.square(pprec * z))
    def costgrad(z):
        return h.T @ (h @ z - y) / (sigma ** 2) + pprec * pprec * z
    for i in range(3):
        xtest = np.random.randn(3)
        cost1 = lin_model.cost(xtest)
        grad1 = lin_model.cost_grad(xtest)
        costref = costfun(xtest)
        gradref = costgrad(xtest)
        assert np.isclose(costref, cost1)
        assert np.isclose(gradref, grad1).all()


def test_linear_model_defaults_lb():
    n = 3
    m = 2
    sigma = 0.1
    h = np.random.randn(m, n)
    x = np.array([1, 2, 2])
    y = h @ x + sigma * np.random.randn(m)
    q = DiagonalOperator(dim=n, s=1 / sigma)
    pprec = np.array([1., 2., 2])
    r = DiagonalOperator(dim=n, s=pprec)
    lin_model = LinearModel(h=h, y=y, q=q, m=np.zeros((n,)), r=r, a=None, b=None, lb=None)
    assert np.all(lin_model.lb <= - np.inf)



