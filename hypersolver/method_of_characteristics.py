""" method of charactersistics
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from hypersolver.derivative import ord1_acc2
from hypersolver.util import term_util, func_util


def moc_next(
    init_vals,
    vars_vals,
    flux_term,
    sink_term,
    time_step,
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
        flux_term:  f
        sink_term:  g
        time_step:

        outputs
        -------
        next_vals:  n

        numerics
        --------
        use scipy.integrate.odeint
    """

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
        dvdt[:] = sink_term - vval*ord1_acc2(flux_term, uval)

        return dydt

    yval0 = np.empty((vars_vals.size + init_vals.size))

    yval0[::2] = vars_vals

    yval0[1::2] = init_vals

    tspan = np.linspace(0, time_step, 10)

    flux_term = term_util(
        func_util(flux_term, init_vals, vars_vals),
        init_vals,
    )

    sink_term = term_util(
        func_util(sink_term, init_vals, vars_vals),
        init_vals,
    )

    results = odeint(_func, yval0, tspan, ml=2, mu=2)

    fill = interp1d(
        results[-1, ::2],
        results[-1, 1::2],
        fill_value=(0.0, 0.0),
        bounds_error=False,
        kind='cubic')

    return fill(vars_vals)
