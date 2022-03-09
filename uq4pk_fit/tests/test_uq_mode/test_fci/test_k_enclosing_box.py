
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from uq4pk_fit.uq_mode.fci.k_enclosing_box import k_enclosing_box


show = False


def test_box_has_right_shape():
    n = 100
    k = 90
    d = 10
    np.random.seed(0)
    test_points = np.random.randn(n, d)
    box = k_enclosing_box(k, test_points)
    # Check that output has the right shape.
    assert box.shape == (2, d)


def test_box_contains_k_points():
    n = 1000
    k = 900
    d = 100
    np.random.seed(0)
    test_points = np.random.randn(n, d)
    box = k_enclosing_box(k, test_points)
    # Count number of points in box.
    mask = np.all(test_points >= box[0], axis=1) & np.all(test_points <= box[1], axis=1)
    points_inside = test_points[mask, :]
    n_inside = points_inside.shape[0]

    assert n_inside == k


def test_visual():
    n = 1000
    k = 900
    d = 2
    np.random.seed(0)
    test_points = np.random.randn(n, d)
    # Apply transformation to distort points
    A = np.array([[3, 0], [0, 1]])
    test_points = test_points @ A
    box = k_enclosing_box(k, test_points)

    # Visualize the box and the points.
    if show:
        ax = plt.axes()
        ax.add_patch(Rectangle(xy=box[0], width=box[1, 0] - box[0, 0], height=box[1, 1] - box[0, 1], fc="none", ec="g",
                               lw=2))
        plt.scatter(x=test_points[:, 0], y=test_points[:, 1], s=4)
        plt.axis("equal")

        plt.show()
