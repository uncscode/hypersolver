""" test: Lax-Friedrichs finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.lax_friedrichs import lx_next, lx_init


def test_lx_init():
    """ test: initialize the array """

    _array = np.array([1., 2., 3., 4., 5.])

    init_array = lx_init(_array)

    assert init_array.shape == (_array.size+2,)

    assert init_array[0] == 0.5 * (_array[1] + _array[0])

    assert init_array[-1] == 0.5 * (_array[-1] + _array[-2])


def test_lx_next():
    """ test: next step according to lx scheme """

    _array = np.linspace(1, 10, 100)

    assert lx_next(
        _array, _array, _array, _array
    ).shape == (_array.size,)

    assert lx_next(
        _array, _array, _array, _array, stability=0.90
    ).shape == (_array.size,)

    assert lx_next(
        _array, _array, 1.0, 0.0
    ).shape == (_array.size,)
