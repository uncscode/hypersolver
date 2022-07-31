""" method of charactersistics
"""
import numpy as np
from scipy.integrate import odeint

from hypersolver.accurate_derivative import acc_derivative


def moc(
    init_vals,
    vars_vals,
    flux_term,
    sink_term,
    time_span,
    **kwargs
):
    """ method of characteristics

        ∂n/∂t + ∂(fn)/∂x = g -->

        above equation can be written as
        ∂xx/∂s = f;           xx(s=0) = x
        ∂nn/∂s = g - n∂f/∂x;  nn(s=0, xx=x) = n

        with solution
        nn(s, xx=x) = n(t, x)

        inputs
        ------
        init_step:  n
        vars_vals:  x
        flux_term:  f
        sink_term:  g
        time_span:

        outputs
        -------
        next_vals:  n(t, x)

        numerics
        --------
        use from scipy.integrate.odeint
    """
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    def _func(yval, sval):
        """ ode function to integrate
        """
        nnval = yval[::2]
        xxval = yval[1::2]

        dydt = np.empty_like(yval)

        dxxdt = dydt[::2]
        dnndt = dydt[1::2]

        dxxdt = flux_term  # noqa: F841
        dnndt = sink_term - nnval * acc_derivative(  # noqa: F841
            flux_term, xxval, 4)

        return dydt

    yval0 = np.empty((vars_vals.size + init_vals.size))
    yval0[::2] = vars_vals
    yval0[1::2] = init_vals
    tval = np.linspace(time_span[0], time_span[-1], 100)

    return odeint(_func, yval0, tval,  ml=2, mu=2)