
import numpy as np

from uq4pk_fit.inference.uq_autodetect import _components_agree, _find_match, _vector_equal


def test_vector_equal():
    resolution = 3
    vec1 = np.array([1, 2])
    vec2 = np.array([2, 3])
    vec3 = np.array([10, 11])
    assert _vector_equal(vec1, vec2, resolution)
    assert not _vector_equal(vec1, vec3, resolution)


def test_find_match():
    resolution = 2
    vec = np.array([1, 2])#
    arr1 = np.array([[1, 2, 4, 2], [5, 6, 6, 2]])
    j1 = _find_match(vec, arr1, resolution)
    assert j1 == 3


def test_find_mismatch():
    resolution = 2
    vec = np.array([1, 2])  #
    arr2 = np.array([[3, 3, 3, 3], [4, 5, 6, 7]])
    j2 = _find_match(vec, arr2, resolution)
    assert j2 is None


def test_components_agree():
    resolution = 2
    arr1 = np.array([[0, 5, 3], [0, 0, 3]])
    arr2 = np.array([[1, 5, 2], [1, 0, 4]])
    assert _components_agree(arr1, arr2, resolution)


def test_components_disagree():
    resolution = 2
    arr1 = np.array([[0, 3, 5], [0, 3, 0]])
    arr3 = np.array([[0, 1, 5], [1, 0, 0]])
    assert not _components_agree(arr1, arr3, resolution)