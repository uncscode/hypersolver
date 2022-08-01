""" 1st derivative, central differencing
"""
import numpy as np
from scipy.misc import central_diff_weights

weights = np.zeros((4, 4))
weights[0, -1:] = np.array([-1/2])
weights[1, -2:] = np.array([1/12, -2/3])
weights[2, -3:] = np.array([-1/60, 3/20, -3/4])
weights[3, -4:] = np.array([1/280, -4/105, 1/5, -4/5])


def _derivative(_func, _xvar, _nacc):
    """ 1st derivative following central finite differencing
    """
    derivative = np.zeros_like(_xvar)
    derivative[0] = (_func[1] - _func[0])/(_xvar[1] - _xvar[0])
    derivative[-1] = (_func[-1] - _func[-2])/(_xvar[-1] - _xvar[-2])
    _weights = central_diff_weights(_nacc+1)

    for nac, idx in zip(range(2, _nacc + 1, 2), range(_nacc//2-1, -1, -1)):
        derivative[nac//2:-nac//2] += (
            _weights[idx]*_func[:-nac] /
            ((_xvar[nac:] - _xvar[:-nac]) / float(nac)) +
            _weights[-idx-1]*_func[nac:] /
            ((_xvar[nac:] - _xvar[:-nac]) / float(nac)))

    if _nacc > 2:
        derivative[:_nacc//2] = np.nan
        derivative[-_nacc//2:] = np.nan

    return derivative


def acc4_derivative(_func, _xvar, _nacc=4):
    """ 1st derivative, central differencing, accuracy up to 4
    """
    derivative = np.zeros_like(_xvar)
    derivative[0] = (_func[1] - _func[0])/(_xvar[1] - _xvar[0])
    derivative[-1] = (_func[-1] - _func[-2])/(_xvar[-1] - _xvar[-2])

    for nac, idx in zip(range(2, _nacc + 1, 2), range(_nacc//2)):

        print(nac//2, -nac//2-1, nac, -nac-1)

        derivative[nac//2] += (
            weights[nac//2-1, -idx-1]*(_func[0] - _func[nac]) /
            ((_xvar[nac] - _xvar[0]) / float(nac)))

        derivative[-nac//2-1] += (
            weights[nac//2-1, -idx-1]*(_func[-nac-1] - _func[-1]) /
            ((_xvar[-1] - _xvar[-nac-1]) / float(nac)))

        derivative[nac//2+1:-nac//2-1] += (
            weights[_nacc//2-1, -idx-1]*(_func[1:-nac-1] - _func[nac+1:-1]) /
            ((_xvar[nac+1:-1] - _xvar[1:-nac-1]) / float(nac)))

    return derivative


def acc_derivative(func, xvar, nacc):
    """ 1st derivative of a _func wrt _xvar with _nacc accuracy

        inputs:
            func: function (np.array)
            xvar: variable (np.array)
            nacc: accuracy (integer)

        outputs:
            derivative: np.array

        NOTES:
            minor bug necessitates repeating calculation if nacc > 2
    """
    if nacc <= 0 or nacc % 2 == 1:
        raise ValueError("n must be positive even")

    if nacc < 6:
        return acc4_derivative(func, xvar, nacc)

    axx_derivative = np.zeros_like(xvar)
    axx_derivative = _derivative(func, xvar, 2)

    for nax in range(4, nacc + 1, 2):
        derivative = _derivative(func, xvar, nax)
        nan_idx = np.isnan(derivative)
        derivative[nan_idx] = axx_derivative[nan_idx]
        axx_derivative = derivative

    return derivative
