""" test: customized simple ode solvers """

from hypersolver.util import xnp as np
from hypersolver.ode_solver import solver_


def test_solver_():
    """ test: solver for ode methods """

    inarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    assert solver_(
        inarr, inarr, (0, 1), 1, method="rk2")[-1].shape == inarr.shape

    assert solver_(
        inarr, 1, (0, 1), 1, method="rk2")[-1].shape == inarr.shape
