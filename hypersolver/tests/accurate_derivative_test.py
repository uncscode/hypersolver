""" test: 1st derivative, central differencing
"""
import numpy as np

from hypersolver.accurate_derivative import acc_derivative


def test_derivative():
    """ test: 1st derivative of a _func wrt _xvar with _nacc accuracy
    """
    func = np.linspace(0, 1, 1000)
    xvar = np.linspace(0, 1, 1000)
    assert acc_derivative(func, xvar, 2).shape == xvar.shape
    assert acc_derivative(func, xvar, 4).shape == xvar.shape
