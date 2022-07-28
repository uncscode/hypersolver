""" Lax-Friedrics scheme """

# import numpy as np


def next_step(
    this_step,
    size_step,
    time_step,
    flux_term,
    sink_term
):
    """ Lax-Friedrics scheme """

    result = this_step.copy()

    result[1:-1] = (
        0.5 * (this_step[2:] + this_step[:-2]) -
        0.5 * (this_step[2:] - this_step[:-2]) *
        flux_term * time_step / size_step +
        sink_term * time_step
    )

    result[0] = (
        0.5 * (this_step[1] + this_step[0]) -
        0.5 * (this_step[1] - this_step[0]) *
        flux_term * time_step / size_step +
        sink_term * time_step
    )

    result[-1] = (
        0.5 * (this_step[-1] + this_step[-2]) -
        0.5 * (this_step[-1] - this_step[-2]) *
        flux_term * time_step / size_step +
        sink_term * time_step
    )

    return result


def solve(
    initial_condition,
    size_step,
    time_step,
    flux_term,
    sink_term
):
    """ Lax-Friedrics scheme """

    result = initial_condition

    for _ in range(1000):
        next_vals = next_step(
            result,
            size_step,
            time_step,
            flux_term,
            sink_term
        )
        result = next_vals

    return result
