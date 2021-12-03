
import pytest
from uq4pk_fit.uq_mode.detection.bump_minimization2 import _discretize_scales, _create_shape_operator


# Testing _discretize_scale.


def test_discretization_coverage():
    min_scale = 0.8
    max_scale = 42
    scales = _discretize_scales(min_scale, max_scale)
    assert min_scale >= scales[0]
    assert max_scale <= scales[-1]


def test_min_must_be_smaller_than_max():
    min_scale = 2
    max_scale = 1.9
    with pytest.raises(AssertionError) as e:
        scales = _discretize_scales(min_scale, max_scale)


# Testing shape operator

def test_shape_operator():
    m = 12
    n = 53
    scales = [1, 1.6, 2.5]
    shape_operator = _create_shape_operator(m=m, n=n, scales=scales)
    assert shape_operator.shape == (3 * len(scales) * m * n, m * n)