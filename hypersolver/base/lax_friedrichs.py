""" Lax-Friedrics finite-difference scheme

    ∂n/∂t + ∂(fn)/∂x = g

    see: hypersolver.base.lax_friedrichs.solver
"""

from hypersolver.base.basic_solver import shared_solver


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
    """ assigning a solver """
    return shared_solver(
        next_step,
        init_vals,
        vars_vals,
        time_span,
        flux_term,
        sink_term,
        **kwargs
    )
