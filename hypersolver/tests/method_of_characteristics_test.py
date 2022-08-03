""" test: method of characteristics """

from hypersolver.util import xnp as np
from hypersolver.method_of_characteristics import moc_next

# pylint: disable=unused-argument
# pylint: disable=unused-variable


def test_moc_next():
    """ test: method of characteristics """

    xvar = np.linspace(1, 10, 1000)

    nvar = 1 * (xvar > 4) - 1 * (xvar > 6)

    def flux_term(yvar, xvar):
        return 1/xvar

    def sink_term(yvar, xvar):
        return -yvar**2

    assert moc_next(
        nvar, xvar, flux_term, sink_term, 0.1).shape == nvar.shape

    assert moc_next(
        nvar, xvar, flux_term, 0.1, 0.1).shape == nvar.shape

    assert moc_next(
        nvar, xvar, 1.0, 0.1, 0.1).shape == nvar.shape
