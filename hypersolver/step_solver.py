""" shared solver between schemes
"""
import os
from hypersolver.util import xnp as np
from hypersolver.util import term_util, func_util, time_step_util
from hypersolver.lax_friedrichs import lx_next
from hypersolver.lax_wendroff import lw_next
if os.environ.get("BACKEND", "numpy") == "jax":
    import jax  # pylint: disable=import-error


def solver_(*args, **kwargs):
    """ set the solver """
    method = kwargs.get("method", "lax_friedrichs")
    if method == "lax_friedrichs":
        next_step = lx_next
    else:
        next_step = lw_next
    if os.environ.get("BACKEND", "numpy") == "jax":
        next_step = jax.jit(next_step)

    def _solver_(  # pylint: disable=too-many-locals
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

        _flux_term = term_util(
            func_util(flux_term, init_vals, vars_vals, **kwargs),
            init_vals)
        _sink_term = term_util(
            func_util(sink_term, init_vals, vars_vals, **kwargs),
            init_vals)

        stability_factor, time_step = (
            stability_factor,
            time_step_util(vars_vals, _flux_term, stability_factor)
        ) if method in [
            "lax_friedrichs", "lax_wendroff"
        ] else (
            np.array((time_span[-1] - time_span[0])/5.0),
            np.array((time_span[-1] - time_span[0])/5.0))

        tidx = np.arange(time_span[0], time_span[-1]+time_step, time_step)
        itrs = 0
        sols = init_vals.reshape(1, -1)

        if method == "lax_wendroff":
            _sink_term = (_sink_term, _sink_term)

        for idx in range(tidx[:-1].size):
            next_vals = next_step(
                sols[itrs, :],
                vars_vals,
                _flux_term,
                _sink_term,
                stability_factor,
            )

            if verbosity == 1:
                print(itrs, idx)

            _flux_term = term_util(
                func_util(flux_term, sols[itrs, :], vars_vals, **kwargs),
                sols[itrs, :])
            _sink_term_ = term_util(
                func_util(sink_term, sols[itrs, :], vars_vals, **kwargs),
                sols[itrs, :])
            if method == "lax_wendroff":
                _sink_term = (_sink_term[1], _sink_term_)
            else:
                _sink_term = _sink_term_

            itrs += 1
            _sols = np.concatenate([sols, next_vals.reshape(1, -1)], axis=0)
            del sols
            sols = _sols

        return sols

    return _solver_(*args, **kwargs)
