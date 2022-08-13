""" Runge-Kutta methods for ode solvers

    dn/dt = f(n, x)

    inputs:
    -------
    init_: n
    vars_: x
    func_: f
    sink_: 0
    time_: Δt

    outputs:
    --------
    next_: n

    numerics:
    ---------
    - rk2 next_: n + Δt*f(n + Δt/2*f(n, x))

"""

from hypersolver.util import jxt as jit
from hypersolver.util import xnp as np
from hypersolver.util import term_util
from hypersolver.util import time_step_util


@jit(nopython=True)
def rk2_next(init_, vars_, func_, sink_, time_):
    """ 2nd order Runge-Kutta method """
    _ = sink_
    step1 = init_ + 0.5 * time_ * term_util(
        func_(init_, vars_), init_)

    return init_ + time_ * term_util(
        func_(step1, vars_), step1)


@jit(nopython=True)
def rk_loop(time, init_, vars_, func_, stability):
    """ loop for rk """
    # pylint: disable=duplicate-code

    time_ = time_step_util(vars_, func_(init_, vars_), stability)

    tidx = np.arange(time[0], time[-1] + time_, time_)

    pts = np.asarray((tidx.size+1., np.asarray(time).size+100., 100.)).min()//1
    sols = np.asarray(init_).reshape(1, -1)
    tims = np.asarray(tidx[0]).reshape(1, -1)

    _yvar = sols[0]

    for itrs in range(tidx[:-1].size):
        next_ = rk2_next(
            _yvar, vars_, func_, 0.0,
            tidx[itrs+1] - tidx[itrs])

        if ((itrs + 1) % (tidx.size//pts)) == 0:
            sols = np.concatenate(
                (sols, np.asarray(next_).reshape(1, -1)), axis=0)
            tims = np.concatenate(
                (tims, np.asarray(tidx[itrs]).reshape(1, -1)), axis=0)

        _yvar = next_

    return tims, sols
