
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.uq_mode import TwoLevelDiscretization, AdaptiveTwoLevelDiscretization


m = 12
n = 53
d1 = 2
d2 = 3
w1 = 1
w2 = 2


def test_two_level_discretization():
    center = np.array([5, 20])
    tld = TwoLevelDiscretization(shape=(m, n), d1=d1, d2=d2, w1=w1, w2=w2, center=center)
    assert tld.dim == m * n
    z1 = 10 * np.ones(tld.n_window)
    z2 = np.random.randn(tld.n_outside)
    z = np.concatenate([z1, z2])
    x = tld.map(z)
    plt.imshow(x.reshape((m, n)))
    plt.show()


def test_adaptive_two_level_discretization():
    atld = AdaptiveTwoLevelDiscretization(shape=(m, n), d1=d1, d2=d2, w1=w1, w2=w2)

