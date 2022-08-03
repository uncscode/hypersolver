""" customized simple ode solvers for use in split methods"""

import os

from hypersolver.util import xnp as np
from hypersolver.util import term_util
from hypersolver.runge_kutta import rk2_next


def solver_(*args, **kwargs):
    """ ode solver """

    method = kwargs.get("method", "rk2")
    nsteps = kwargs.get("nsteps")

    if method == "rk2":
        next_step = rk2_next
    else:
        raise ValueError("method not supported")

    def _solver_(init_vals, vars_vals, time_span, func_term, **kwargs):
        """ solver for the ode methods

            function to loop over `time_step`s using ode schemes

            dn/dt = f(n, x)

            inputs:
            -------
            init_vals: n
            vars_vals: x
            time_span: (start, end)
            func_term: f

            outputs:
            --------
            sols: n (t, x)

            methods:
            --------
            - rk2:  hypersolver.runge_kutta.rk2_next
        """

        vars_vals = term_util(vars_vals, init_vals)

        time_step = np.max(
            np.array([
                1e-5,
                0.01*np.min(np.array(
                    [time_span[-1] - time_span[0], 1]))
            ])
        ) if nsteps is None else np.array(time_span[-1] - time_span[0])/nsteps

        tidx = np.arange(time_span[0], time_span[-1] + time_step, time_step)
        itrs = 0

        sols = init_vals.reshape(1, -1)

        for _ in range(tidx[:-1].size):

            next_vals = next_step(
                sols[itrs, :], vars_vals, func_term, time_step, **kwargs)

            if os.environ.get("HS_VERBOSITY", "0") == "1":
                print(itrs)

            itrs += 1

            sols = np.concatenate((sols, next_vals.reshape(1, -1)), axis=0)

        return sols

    return _solver_(*args, **kwargs)
