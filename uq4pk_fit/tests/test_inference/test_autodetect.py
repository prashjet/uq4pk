
from matplotlib import pyplot as plt
import numpy as np

from uq4pk_fit.inference.uq_autodetect import _components_agree, _detect_components


def test_components_dont_match():
    scale = 3.
    upper = np.loadtxt("upper3.csv", delimiter=",")
    lower = np.loadtxt("lower3.csv", delimiter=",")
    treshold = upper.max() * 0.01
    components1 = _detect_components(upper, scale=scale, treshold=treshold)
    components2 = _detect_components(lower, scale=scale, treshold=treshold)
    assert not _components_agree(components1, components2, scale=scale)

def test_components_match():
    scale = 4.
    upper = np.loadtxt("upper4.csv", delimiter=",")
    lower = np.loadtxt("lower4.csv", delimiter=",")
    vmax = upper.max()
    treshold = vmax * 0.01
    components1 = _detect_components(lower, scale=scale, treshold=treshold)
    components2 = _detect_components(upper, scale=scale, treshold=treshold)
    assert _components_agree(components1, components2, scale=scale)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(lower, vmax=vmax, vmin=0.)
    for i in range(components1.shape[1]):
        x_coord = components1[1, i]
        y_coord = components1[0, i]
        ax[0].add_patch(plt.Circle((x_coord, y_coord), scale, color='lime',
                                   fill=False))
    ax[1].imshow(upper, vmax=vmax, vmin=0.)
    for i in range(components1.shape[1]):
        x_coord = components2[1, i]
        y_coord = components2[0, i]
        ax[1].add_patch(plt.Circle((x_coord, y_coord), scale, color='lime',
                                   fill=False))
    plt.show()

