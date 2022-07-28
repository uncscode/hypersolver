""" test: Lax-Friedrics scheme """

import numpy as np

from hypersolver.lax_friedrichs import next_step, solver


def test_next_step():
    """ test: next step according to Lax-Friedrics scheme """

    assert next_step(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
        1,
        np.array([1, 2, 3]),
        np.array([1, 2, 3])
    ).shape == np.array([1, 2, 3]).shape


def test_solver():
    """ test: solver accorrding to Lax-Friedrics scheme """

    assert solver(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
        (0, 1),
        1,
        0
    )[0].shape == np.array([1, 2, 3]).shape

    assert solver(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
        (0, 1),
        np.array([1, 2, 3]),
        np.array([1, 2, 3])
    )[0].shape == np.array([1, 2, 3]).shape

    assert solver(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
        (0, 5),
        lambda yval, xval, **kwargs: 1,
        lambda yval, xval, **kwargs: 0,
        verbosity=1
    )[0].shape == np.array([1, 2, 3]).shape
