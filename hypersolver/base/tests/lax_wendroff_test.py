""" test: Lax-Friedrics scheme """

import numpy as np

from hypersolver.base.lax_wendroff import next_step, solver

_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_next_step():
    """ test: next step according to Lax-Friedrics scheme """
    assert next_step(
        _array,
        _array,
        1,
        _array,
        _array
    ).shape == _array.shape


def test_solver():
    """ test: solver accorrding to Lax-Friedrics scheme """

    assert solver(
        _array,
        _array,
        (0, 1),
        1,
        0
    )[0].shape == _array.shape

    assert solver(
        _array,
        _array,
        (0, 1),
        _array,
        _array
    )[0].shape == _array.shape

    assert solver(
        _array,
        _array,
        (0, 5),
        lambda yval, xval, **kwargs: 1,
        lambda yval, xval, **kwargs: 0,
        verbosity=1
    )[0].shape == _array.shape
