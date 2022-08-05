""" Runge-Kutta methods for ode solvers """

from hypersolver.util import term_util
from hypersolver.util import jxt as jit


def rk2_next(
    init_vals,
    vars_vals,
    func_term,
    time_step,
    **kwargs
):
    """ 2nd order Runge-Kutta method

        dn/dt = f(n, x)

        inputs:
        -------
        init_vals: n
        vars_vals: x
        func_term: f
        time_step: Δt

        outputs:
        --------
        next_vals: n

        numerics:
        ---------
        - next_vals: n + Δt*f(n + Δt/2*f(n, x))
    """

    _ = kwargs.get("method", "rk2")

    if callable(func_term):
        _func_term = jit(nopython=True)(func_term)
    else:
        @jit(nopython=True)
        def _func_term_(yvar, xvar):  # pylint: disable=unused-argument
            """ jit the function """
            return func_term
        _func_term = _func_term_

    @jit(nopython=True)
    def _next(init_vals, vars_vals, func_term, time_step):
        """ 2nd order Runge-Kutta method """
        step1 = init_vals + 0.5 * time_step * term_util(
            func_term(init_vals, vars_vals), init_vals)

        return init_vals + time_step * term_util(
            func_term(step1, vars_vals), step1)

    return _next(init_vals, vars_vals, _func_term, time_step)
