""" test: Lax-Friedrics scheme """

import numpy as np

from hypersolver.method_of_characteristics import moc_next

xvars = np.linspace(1, 10, 100)
yvars = 1 * (xvars > 4) - 1 * (xvars > 6)
flux_term = np.ones_like(xvars)
sols = moc_next(
    yvars, xvars, flux_term, 0, 0.1,
)


def test_moc():
    """ test: method of characteristics """

    assert sols.shape == xvars.shape
