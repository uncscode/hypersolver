""" test: Lax-Friedrics scheme """

import numpy as np

from hypersolver import select_solver
solver = select_solver("lax_wendroff")

_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


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
