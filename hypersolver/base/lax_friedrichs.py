""" Lax-Friedrics finite-difference scheme

    ∂n/∂t + ∂(fn)/∂x = g

    see: hypersolver.lax_friedrichs.solver
"""

import numpy as np

from hypersolver.util import term_util


def next_step(
    this_step,
    vars_vals,
    time_step,
    flux_term,
    sink_term
):
    """ next step according to Lax-Friedrics finite-difference scheme
    """

    result = this_step.copy()

    result[1:-1] = (
        0.5 * (this_step[2:] + this_step[:-2]) -
        1.0 * (
            this_step[2:]*flux_term[2:] -
            this_step[:-2]*flux_term[:-2]
        ) * time_step / (vars_vals[2:] - vars_vals[:-2]) +
        sink_term[1:-1] * time_step
    )

    result[0] = (
        0.5 * (this_step[1] + this_step[0]) -
        1.0 * (
            this_step[1]*flux_term[1] - this_step[0]*flux_term[0]
        ) * time_step / (vars_vals[1] - vars_vals[0]) +
        sink_term[0] * time_step
    )

    result[-1] = (
        0.5 * (this_step[-1] + this_step[-2]) -
        1.0 * (
            this_step[-1]*flux_term[-1] -
            this_step[-2]*flux_term[-2]
        ) * time_step / (vars_vals[-1] - vars_vals[-2]) +
        sink_term[-1] * time_step
    )

    return result


def solver(
    init_vals,
    vars_vals,
    time_span,
    flux_term,
    sink_term,
    **kwargs
):
    """ solver accorrding to Lax-Friedrics finite-difference scheme

        equation:   ∂n/∂t + ∂(fn)/∂x = g

        init_vals:  initial  values of n (np.array)
        vars_vals:  variable values of x (np.array)
        time_span:  time span of t (list or tuple)
        flux_term:  flux term, f (either explicit or function)
        sink_term:  sink term, g (either explicit or function)

        additional keyword arguments:
            - stability_factor (float, 0.98): factor of stability (λ)
            - verbosity (int, 0): verbosity printing level

        returns:
        np.array(shape(_time.size, vars_vals.size))

        notes:
        flux_term and sink_term can be either explicit or functions;
        if functions, they must be defined as: function(n, x, **kwargs)

        numerics (letting i be vars_vals index, j be time_span index):

        n(j+1, i) = (
            0.5 * (n(j, i+1) + n(j+1, i-1)) -
            1.0 * (
                n(j, x+1)*f(n(t, i+1)) -
                n(j, x-1)*f(n(t, i-1))
            ) * time_step / (x(i+1) - x(i-1)) +
            g(j, i) * time_step

        time_step = (
            stability_factor *
            (x(i+1) - x(i-1)).min() /
            (fn(j=0, i)).max()
        )

    """

    stability_factor = kwargs.get('stability_factor', 0.98)
    verbosity = kwargs.get('verbosity', 0)

    vars_vals = term_util(vars_vals, init_vals)

    if isinstance(flux_term, type(next_step)):
        flux_term = flux_term(init_vals, vars_vals, **kwargs)
    if isinstance(sink_term, type(next_step)):
        sink_term = sink_term(init_vals, vars_vals, **kwargs)
    flux_term = term_util(flux_term, init_vals)
    sink_term = term_util(sink_term, init_vals)

    time_step = (
        stability_factor *
        np.diff(vars_vals).min() /
        np.abs(flux_term).max()
    )

    tidx = np.arange(time_span[0], time_span[-1]+time_step, time_step)
    sols = np.zeros((tidx.size, init_vals.size))
    itrs = 0
    sols[itrs] = init_vals

    for idx in tidx[:-1]:
        next_vals = next_step(
            sols[itrs],
            vars_vals,
            time_step,
            flux_term,
            sink_term
        )

        if verbosity == 1:
            print(idx, itrs)

        if isinstance(flux_term, type(next_step)):
            flux_term = flux_term(sols[itrs], vars_vals, **kwargs)
        if isinstance(sink_term, type(next_step)):
            sink_term = sink_term(sols[itrs], vars_vals, **kwargs)
        flux_term = term_util(flux_term, sols[itrs])
        sink_term = term_util(sink_term, sols[itrs])

        itrs += 1
        sols[itrs] = next_vals

    return sols
