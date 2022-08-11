""" test: Runge-Kutta methods for ode solvers """

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.runge_kutta import rk2_next


def test_rk2_next():
    """ test: 2nd order Runge-Kutta method """

    inputs = np.linspace(1, 100, 1000)

    @jit(nopython=True)
    def func(yvar, xvar):
        """ func """
        return yvar/xvar

    assert rk2_next(
        inputs, inputs, func, 0.1).shape == inputs.shape
