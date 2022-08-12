""" test: Runge-Kutta methods for ode solvers """

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.runge_kutta import rk2_next, rk_loop


def test_rk2_next():
    """ test: 2nd order Runge-Kutta method """

    inputs = np.linspace(1, 100, 1000)

    @jit(nopython=True)
    def func(yvar, xvar):
        """ func """
        return yvar/xvar

    assert rk2_next(
        inputs, inputs, func, 0.0, 0.1).shape == inputs.shape


def test_lx_loop():
    """ test: loop for lx scheme """

    xvar = np.linspace(1, 10, 100)
    yvar = 1.0 * (xvar > 4) - 1.0 * (xvar > 6)

    @jit(nopython=True)
    def flux(yvar, xvar):  # pylint: disable=unused-argument
        """ flux """
        return 5 / xvar

    time = np.linspace(0, 2, 1000)
    assert (rk_loop(time, yvar.reshape(1, -1), xvar,
            flux, 0.9))[-1].shape[0] >= 100
