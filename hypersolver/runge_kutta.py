""" Runge-Kutta methods for ode solvers

    dn/dt = f(n, x)

    inputs:
    -------
    init_: n
    vars_: x
    func_: f
    time_: Δt

    outputs:
    --------
    next_: n

    numerics:
    ---------
    - next_vals: n + Δt*f(n + Δt/2*f(n, x))

"""

from hypersolver.util import jxt as jit
from hypersolver.util import term_util


@jit(nopython=True)
def rk2_next(init_, vars_, func_, time_):
    """ 2nd order Runge-Kutta method """

    step1 = init_ + 0.5 * time_ * term_util(
        func_(init_, vars_), init_)

    return init_ + time_ * term_util(
        func_(step1, vars_), step1)
