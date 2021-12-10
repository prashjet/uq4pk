
import pytest
from uq4pk_fit.uq_mode.blob_detection.scale_space_minimization import _discretize_scales


# Testing _discretize_scale.


def test_discretization_coverage():
    min_scale = 0.8
    max_scale = 42
    scales = _discretize_scales(min_scale, max_scale, num_scale=13)
    assert min_scale >= scales[0]
    assert max_scale <= scales[-1]


def test_min_must_be_smaller_than_max():
    min_scale = 2
    max_scale = 1.9
    with pytest.raises(AssertionError) as e:
        scales = _discretize_scales(min_scale, max_scale, num_scale=13)