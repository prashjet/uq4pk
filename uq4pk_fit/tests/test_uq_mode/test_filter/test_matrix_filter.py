"""
Test for "matrix_filter.py"
"""

import numpy as np

from uq4pk_fit.uq_mode import MatrixFilter


def test_matrix_filter1():
    m = 20
    n = 10
    mat = np.array([[1, 2, 1],
                    [1, 3, 1],
                    [1, 3, 1],
                    [1, 2, 1]])
    center = np.array([1, 1]) # center at the first "3".
    # Create MatrixFilter object, where mat is positioned at [0, 0]
    position = np.array([0, 0])
    matrix_filter = MatrixFilter(m=m, n=n, position=position, mat=mat, center=center)

    # assert that the weights and the indices are correct:
    assert np.all(matrix_filter.weights == np.array([3, 1, 3, 1, 2, 1]))


def test_matrix_filter2():
    m = 17
    n = 19
    mat = np.array([[1, 2, 1],
                    [1, 3, 1],
                    [1, 3, 1],
                    [1, 2, 1]])
    center = np.array([1, 1])
    position = np.array([16, 18])
    matrix_filter = MatrixFilter(m, n, position, mat, center)
    assert np.all(matrix_filter.weights == np.array([[1, 2, 1, 3]]))

#test_matrix_filter1()
test_matrix_filter2()