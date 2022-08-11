""" test: Lax-Wendroff finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.lax_wendroff import lw_next, lw_loop


def test_lx_next():
    """ test: next step according to lw scheme """

    _array = np.linspace(10, 20, 1000)

    assert lw_next(
        _array, _array, _array, (_array, _array), 0.01
    ).shape == (_array.size,)

    assert lw_next(
        _array, _array, _array, (_array, _array), 0.01
    ).shape == (_array.size,)

    assert lw_next(
        _array, _array, _array, (_array, _array), 0.09
    ).shape == (_array.size,)


def test_lw_loop():
    """ test: loop for lw scheme """

    xvar = np.linspace(1, 10, 100)
    yvar = 1.0 * (xvar > 4) - 1.0 * (xvar > 6)

    @jit(nopython=True)
    def flux(yvar, xvar):  # pylint: disable=unused-argument
        """ flux """
        return 5 / xvar

    @jit(nopython=True)
    def sink(yvar, xvar):  # pylint: disable=unused-argument
        """ sink """
        return -0.01 * yvar

    time = np.linspace(0, 2, 1000)
    assert (lw_loop(time, yvar.reshape(1, -1), xvar,
            flux, sink, 0.9))[-1].shape[0] >= 100
