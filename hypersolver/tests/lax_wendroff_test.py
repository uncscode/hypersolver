""" test: Lax-Wendroff finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.lax_wendroff import lw_next


def test_lx_next():
    """ test: next step according to lw scheme """

    _array = np.linspace(10, 20, 1000)

    assert lw_next(
        _array, _array, _array, (_array, _array)
    ).shape == (_array.size,)

    assert lw_next(
        _array, _array, _array, _array
    ).shape == (_array.size,)

    assert lw_next(
        _array, _array, _array, (_array, _array), stability=0.90
    ).shape == (_array.size,)

    assert lw_next(
        _array, _array, 1.0, 0.0, stability=0.95
    ).shape == (_array.size,)

    assert lw_next(
        _array, _array, 1.0, (0.0, 0.0), stability=0.95
    ).shape == (_array.size,)
