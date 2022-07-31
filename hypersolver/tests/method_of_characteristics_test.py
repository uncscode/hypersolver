""" test: Lax-Friedrics scheme """

import numpy as np

from hypersolver import select_solver
moc = select_solver("method_of_characteristics")

xvars = np.linspace(1, 10, 100)
yvars = 1 * (xvars > 4) - 1 * (xvars > 6)
flux_term = np.ones_like(xvars)
sols_moc_x, sols_moc_y = moc(
    yvars, xvars, (0, 1), flux_term, 0
)


def test_moc():
    """ test: method of characteristics """

    sols_moc_x.shape[-1] == xvars.shape[0]
    sols_moc_y.shape[-1] == yvars.shape[0]
