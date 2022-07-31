""" shared solver between schemes
"""
import numpy as np

from hypersolver.util import term_util
from hypersolver.lax_friedrichs import lx_next
from hypersolver.lax_wendroff import lw_next


def solver(method):
    """ set the solver """
    if method == "lax_friedrichs":
        next_step = lx_next
    if method == "lax_wendroff":
        next_step = lw_next

    def _solver(  # pylint: disable=too-many-arguments
        init_vals,
        vars_vals,
        time_span,
        flux_term,
        sink_term,
        **kwargs
    ):
        """ solver accorrding to finite-difference schemes """

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
                flux_term,
                sink_term,
                stability_factor,
            )

            if verbosity == 1:
                print(itrs, idx)

            if isinstance(flux_term, type(next_step)):
                flux_term = flux_term(sols[itrs], vars_vals, **kwargs)
            if isinstance(sink_term, type(next_step)):
                sink_term = sink_term(sols[itrs], vars_vals, **kwargs)
            flux_term = term_util(flux_term, sols[itrs])
            sink_term = term_util(sink_term, sols[itrs])

            itrs += 1
            sols[itrs] = next_vals

        return sols

    return _solver
