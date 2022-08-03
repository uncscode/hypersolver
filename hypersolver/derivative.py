""" calculating derivatives central differencing

    available functions:
        - ord1_acc2: order=1, accuracy=2
        - ord2_acc2: order=2, accuracy=2
"""

from hypersolver.util import xnp as np


def ord1_acc2(_func, _xvar):
    """ central differencing: order=1, accuracy=2 """

    _func, _xvar = np.array(_func), np.array(_xvar)

    _derivative = (
        _func[2:] - _func[:-2]
    )/(_xvar[2:] - _xvar[:-2])

    return np.pad(
        _derivative,
        (1, 1),
        mode='constant',
        constant_values=(
            (_func[1] - _func[0])/(_xvar[1] - _xvar[0]),
            (_func[-1] - _func[-2])/(_xvar[-1] - _xvar[-2])
        ),)


def ord2_acc2(_func, _xvar):
    """ central differencing: order=2, accuracy=2 """

    _func, _xvar = np.array(_func), np.array(_xvar)

    _derivative = (
        _func[2:] - 2.0*_func[1:-1] + _func[:-2]
    )/((_xvar[2:] - _xvar[:-2])/2.0)**2

    return np.pad(
        _derivative,
        (1, 1),
        mode='constant',
        constant_values=(
            (_func[2] - 2.0*_func[1] + _func[0])/(_xvar[1] - _xvar[0])**2,
            (_func[-1] - 2.0*_func[-2] + _func[-3])/(_xvar[-1] - _xvar[-2])**2
        ),)
