""" test basic solver
"""

from hypersolver import solver
from hypersolver.util import xnp as np

_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_solver_lax_():
    """ test step_solver: lax_friedrichs and lax_wendroff """
    for method in [
        "lax_friedrichs", "lax_wendroff",
    ]:
        assert solver(
            _array,
            _array,
            (0, 1),
            1,
            0,
            method=method,
        )[0].shape == _array.shape

        assert solver(
            _array,
            _array,
            (0, 1),
            _array,
            _array,
            method=method,
        )[0].shape == _array.shape

        assert solver(
            _array,
            _array,
            (0, 5),
            lambda yval, xval, **kwargs: 1,
            lambda yval, xval, **kwargs: 0,
            verbosity=1,
            method=method,
        )[0].shape == _array.shape
