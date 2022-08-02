""" Runge-Kutta methods for ode solvers """

from hypersolver.util import term_util, func_util


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

    step1 = init_vals + 0.5 * time_step * term_util(
        func_util(func_term, init_vals, vars_vals, **kwargs),
        init_vals)

    return init_vals + time_step * term_util(
        func_util(func_term, step1, vars_vals, **kwargs),
        step1)
