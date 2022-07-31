""" test: 1st derivative, central differencing
"""
import numpy as np

from hypersolver.accurate_derivative import acc_derivative


def test_derivative_shape():
    """ test: 1st derivative of a _func wrt _xvar with _nacc accuracy
    """
    func = np.linspace(0, 1, 1000)
    xvar = np.linspace(0, 1, 1000)
    assert acc_derivative(func, xvar, 2).shape == xvar.shape
    assert acc_derivative(func, xvar, 4).shape == xvar.shape


def test_derivative_value():
    """ test: 1st derivative of a _func wrt _xvar with _nacc accuracy
    """
    xvar = np.linspace(1, 5, 1000)
    assert np.abs(
        acc_derivative(np.sin(xvar), xvar, 2) - np.cos(xvar)
    ).sum() > np.abs(
        acc_derivative(np.sin(xvar), xvar, 4) - np.cos(xvar)
    ).sum()
    assert np.abs(
        acc_derivative(1/xvar, xvar, 2) + 1/xvar**2
    ).sum() > np.abs(
        acc_derivative(1/xvar, xvar, 10) + 1/xvar**2
    ).sum()
