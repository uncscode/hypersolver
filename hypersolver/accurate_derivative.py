""" nth derivative
"""
import numpy as np
from scipy.misc import central_diff_weights


def acc_derivative(func, xvar, nacc):
    """
    Calculates the nth derivative of a function

    Args:
        func: function
        xvar: variable
        nacc: accuracy

    Returns:
        derivative: array

    FIXME: prevent carrying edges, temporary workaround:
        test2 = nth_derivative(y,x,2)
        test4 = nth_derivative(y,x,4)
        test6 = nth_derivative(y,x,6)
        test8 = nth_derivative(y,x,8)

        test4[np.isnan(test4)] = test2[np.isnan(test4)]
        test6[np.isnan(test6)] = test4[np.isnan(test6)]
        test8[np.isnan(test8)] = test6[np.isnan(test8)]
    """
    if nacc == 0 or nacc % 2 == 1:
        raise ValueError("n must be positive even")

    derivative = np.zeros_like(xvar)
    derivative[0] = (func[1] - func[0])/(xvar[1] - xvar[0])
    derivative[-1] = (func[-1] - func[-2])/(xvar[-1] - xvar[-2])
    weights = central_diff_weights(nacc+1)

    for nac, idx in zip(range(2, nacc + 1, 2), range(nacc//2-1, -1, -1)):
        # fixme: prevent derivative from applying current weights on edges
        # instead, take previous weights for edges
        derivative[nac//2:-nac//2] += (
            weights[idx]*func[:-nac ]/((xvar[nac:] - xvar[:-nac])/float(nac)) +
            weights[-idx-1]*func[ nac:]/((xvar[nac:] - xvar[:-nac])/float(nac))
        )
    if nacc > 2:
        derivative[:nacc//2] = np.nan
        derivative[-nacc//2:] = np.nan
    return derivative
