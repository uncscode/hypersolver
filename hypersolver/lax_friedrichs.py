""" Lax-Friedrics scheme """

import numpy as np

from hypersolver.util import term_util


def next_step(
    this_step,
    vars_vals,
    time_step,
    flux_term,
    sink_term
):
    """ Lax-Friedrics scheme """

    result = this_step.copy()

    result[1:-1] = (
        0.5 * (this_step[2:] + this_step[:-2]) -
        1.0 * (this_step[2:] - this_step[:-2]) *
        flux_term[1:-1] * time_step / (vars_vals[2:] - vars_vals[:-2]) +
        sink_term[1:-1] * time_step
    )

    result[0] = (
        0.5 * (this_step[1] + this_step[0]) -
        1.0 * (this_step[1] - this_step[0]) *
        flux_term[0] * time_step / (vars_vals[1] - vars_vals[0]) +
        sink_term[0] * time_step
    )

    result[-1] = (
        0.5 * (this_step[-1] + this_step[-2]) -
        1.0 * (this_step[-1] - this_step[-2]) *
        flux_term[-1] * time_step / (vars_vals[-1] - vars_vals[-2]) +
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
    """ Lax-Friedrics scheme """

    stability_factor = kwargs.get('stability_factor', 0.8)
    verbosity = kwargs.get('verbosity', 0)

    vars_vals = term_util(vars_vals, init_vals)

    if isinstance(flux_term, type(next_step)):
        flux_term = flux_term(init_vals, vars_vals, **kwargs)
    if isinstance(sink_term, type(next_step)):
        sink_term = flux_term(init_vals, vars_vals, **kwargs)
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
            sink_term = flux_term(sols[itrs], vars_vals, **kwargs)
        flux_term = term_util(flux_term, sols[itrs])
        sink_term = term_util(sink_term, sols[itrs])

        itrs += 1
        sols[itrs] = next_vals

    return sols
