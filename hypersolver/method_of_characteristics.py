""" method of charactersistics
"""
import numpy as np
from scipy.integrate import odeint

from hypersolver.accurate_derivative import acc_derivative


def moc(
    init_vals,
    vars_vals,
    time_span,
    flux_term,
    sink_term,
    **kwargs
):
    """ method of characteristics

        ∂n/∂t + ∂(fn)/∂x = g

        above equation can be written as
        ∂xx/∂s = f;           xx(s=0) = x
        ∂nn/∂s = g - n∂f/∂x;  nn(s=0, xx=x) = n

        with solution
        nn(s, xx=x) = n(t, x)

        inputs
        ------
        init_step:  n
        vars_vals:  x
        time_span:
        flux_term:  f
        sink_term:  g

        outputs
        -------
        next_vars:  x
        next_vals:  n

        numerics
        --------
        use from scipy.integrate.odeint
    """
    _ = kwargs

    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    def _func(yval, tval):
        """ ode function to integrate
        """
        uval = yval[::2]
        vval = yval[1::2]

        dydt = np.empty_like(yval)

        dudt = dydt[::2]
        dvdt = dydt[1::2]

        dudt[:] = flux_term  # noqa: F841
        dvdt[:] = sink_term - vval*acc_derivative(
            flux_term, uval, 4)

        return dydt

    yval0 = np.empty((vars_vals.size + init_vals.size))
    yval0[::2] = vars_vals
    yval0[1::2] = init_vals
    tspan = np.linspace(time_span[0], time_span[-1], 100)
    results = odeint(_func, yval0, tspan, ml=2, mu=2)

    return (results[:, ::2], results[:, 1::2])
