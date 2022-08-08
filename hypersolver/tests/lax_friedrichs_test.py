""" test: Lax-Friedrichs finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.lax_friedrichs import lx_next, lx_loop


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


def test_lx_loop():
    """ test: loop for lx scheme """

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
    assert (lx_loop(time, yvar.reshape(1, -1), xvar,
            flux, sink, 0.9))[-1].shape[0] >= 100
