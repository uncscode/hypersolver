""" test: Lax-Friedrics scheme """

import numpy as np

from hypersolver import select_solver

solver = select_solver("lax_friedrichs")


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
