""" test: Runge-Kutta methods for ode solvers """

from hypersolver.util import xnp as np
from hypersolver.runge_kutta import rk2_next


def test_rk2_next():
    """ test: 2nd order Runge-Kutta method """

    inputs = np.linspace(1, 100, 1000)

    assert rk2_next(
        inputs, inputs, inputs, 0.1).shape == inputs.shape

    assert rk2_next(
        inputs, inputs, 1, 0.1).shape == inputs.shape

    assert rk2_next(
        inputs, 1, 1, 0.1).shape == inputs.shape
