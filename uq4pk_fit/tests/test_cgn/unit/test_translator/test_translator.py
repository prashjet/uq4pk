
import numpy as np
import pytest

from uq4pk_fit.cgn.translator import Translator
from uq4pk_fit.cgn.cnls_solve import CNLS, NullConstraint

from ..test_problem.problem_fixtures import three_parameter_problem, unconstrained_problem


def test_translate(three_parameter_problem):
    # Initialize translator
    translator = Translator(three_parameter_problem)
    # Translate dummy problem
    dummy_cnls = translator.translate()
    assert dummy_cnls.dim == three_parameter_problem.n
    assert isinstance(dummy_cnls, CNLS)


def test_modify_function(three_parameter_problem):
    translator = Translator(three_parameter_problem)
    cnls = translator.translate()
    testfun = three_parameter_problem.fun
    modified_fun = translator._modify_function(testfun)
    # Check that modified function can be called with concatenated vector
    x_conc = np.random.randn(cnls.dim)
    y = modified_fun(x_conc)
    m = three_parameter_problem.m
    assert y.size == m


def test_translate_constraints(three_parameter_problem):
    translator = Translator(three_parameter_problem)
    translated_eqcon = translator._combine_constraints("eq")
    assert np.isclose(translated_eqcon.a[:, :-1], three_parameter_problem.constraints[0].a).all()
    translated_incon = translator._combine_constraints("ineq")
    n1 = three_parameter_problem.shape[0]
    assert np.isclose(translated_incon.a[:, n1:-1], three_parameter_problem.constraints[1].a).all()


def test_translate_unconstrained(unconstrained_problem):
    translator = Translator(unconstrained_problem)
    eqcon = translator._combine_constraints("eq")
    incon = translator._combine_constraints("ineq")
    assert isinstance(eqcon, NullConstraint)
    assert isinstance(incon, NullConstraint)


