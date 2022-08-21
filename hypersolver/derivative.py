""" calculating derivatives central differencing

    available functions:
        - ord1_acc2: order=1, accuracy=2
        - ord1_acc4: order=1, accuracy=4
        - ord2_acc2: order=2, accuracy=2
        - ord2_acc4: order=2, accuracy=4
"""

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit


@jit(nopython=True)
def ord1_acc2(_func, _xvar):
    """ central differencing: order=1, accuracy=2 """

    _func, _xvar = np.asarray(_func), np.asarray(_xvar)

    _derivative = (
        - (1/2)*_func[:-2] + (1/2)*_func[2:]
    )/((_xvar[2:] - _xvar[:-2])/2)

    _derivative0 = np.asarray(
        [(_func[1] - _func[0])/(_xvar[1] - _xvar[0])]
    )
    _derivative1 = np.asarray(
        [(_func[-1] - _func[-2])/(_xvar[-1] - _xvar[-2])]
    )

    return np.concatenate((_derivative0, _derivative, _derivative1))


@jit(nopython=True)
def ord1_acc4(_func, _xvar):
    """ central differencing: order1, accuracy=4 """

    _func, _xvar = np.asarray(_func), np.asarray(_xvar)

    _result1 = (
        - (2/3)*_func[:-2] + (2/3)*_func[2:]
    )/((_xvar[2:] - _xvar[:-2])/2)

    _result2 = np.concatenate((
        -np.asarray([_result1[0] - _result1[0]*3/4]),
        (
            (1/12)*_func[:-4] - (1/12)*_func[4:]
        )/((_xvar[4:] - _xvar[:-4])/4),
        -np.asarray([_result1[-1] - _result1[-1]*3/4]),
    ))

    _derivative = _result1 + _result2

    _derivative0 = np.asarray(
        [(_func[1] - _func[0])/(_xvar[1] - _xvar[0])]
    )
    _derivative1 = np.asarray(
        [(_func[-1] - _func[-2])/(_xvar[-1] - _xvar[-2])]
    )

    return np.concatenate(
        (_derivative0, _derivative, _derivative1)
    )


@jit(nopython=True)
def ord2_acc2(_func, _xvar):
    """ central differencing: order=2, accuracy=2 """

    _func, _xvar = np.asarray(_func), np.asarray(_xvar)

    _derivative = (
        _func[2:] - 2.0*_func[1:-1] + _func[:-2]
    )/((_xvar[2:] - _xvar[:-2])/2.0)**2

    _derivative0 = np.asarray(
        [(_func[2] - 2.0*_func[1] + _func[0])/(_xvar[1] - _xvar[0])**2]
    )
    _derivative1 = np.asarray(
        [(_func[-1] - 2.0*_func[-2] + _func[-3])/(_xvar[-1] - _xvar[-2])**2]
    )

    return np.concatenate((_derivative0, _derivative, _derivative1))


@jit(nopython=True)
def ord2_acc4(_func, _xvar):
    """ central differencing: order=2, accuracy=4 """

    _func, _xvar = np.asarray(_func), np.asarray(_xvar)

    _result10 = (
        _func[:-2] - 2.0*_func[1:-1] + _func[2:]
    )/((_xvar[2:] - _xvar[:-2])/2.0)**2

    _result20 = (
        (4/3)*_func[:-2] - (5/2)*_func[1:-1] + (4/3)*_func[2:]
    )/((_xvar[2:] - _xvar[:-2])/2.0)**2

    _result2 = np.concatenate((
        np.array([_result10[0]]),
        (
            -(1/12)*_func[:-4] - (1/12)*_func[4:]
        )/((_xvar[4:] - _xvar[:-4])/4.0)**2 + _result20[1:-1],
        np.array([_result10[-1]]),
    ))

    _derivative = _result2

    _derivative0 = np.asarray(
        [(_func[2] - 2.0*_func[1] + _func[0])/(_xvar[1] - _xvar[0])**2]
    )
    _derivative1 = np.asarray(
        [(_func[-1] - 2.0*_func[-2] + _func[-3])/(_xvar[-1] - _xvar[-2])**2]
    )

    return np.concatenate(
        (_derivative0, _derivative, _derivative1)
    )
