""" test: Lax-Friedrichs finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.lax_friedrichs import lx_next


def test_lx_next():
    """ test: next step according to lx scheme """

    _array = np.linspace(1, 10, 100)

    assert lx_next(
        _array, _array, _array, _array, 0.01
    ).shape == (_array.size,)

    assert lx_next(
        _array, _array, _array, _array, 0.001
    ).shape == (_array.size,)

    assert lx_next(
        _array, _array, 1.0, 0.0, 0.01
    ).shape == (_array.size,)
