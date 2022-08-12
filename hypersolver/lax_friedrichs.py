""" Lax-Friedrics finite-difference scheme

    ∂n/∂t + ∂(fn)/∂x = g

    inputs
    ------
    init_:  n
    vars_:  x
    flux_:  f
    sink_:  g
    time_:  Δt

    outputs
    -------
    next_:  n

    numerics
    --------
    n(j+1, i) = n(j, i) + Δt (g - Δ(fn)/Δx)(j, i)

    Δt ≤ λΔx/f ∀ x
    Δ(fn)/Δx is first-order derivative with accuracy of 2
    n(j, i) = (n(j, i-1) + n(j, i+1))/2

"""

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.util import time_step_util

from hypersolver.derivative import ord1_acc2


@jit(nopython=True)
def lx_next(init_, vars_, flux_, sink_, time_):
    """ next step according to Lax-Friedrics finite-difference scheme """

    _init_ = np.concatenate((
        np.asarray([0.5 * (init_[1] + init_[0])]),
        np.asarray((0.5 * (init_[2:] + init_[:-2]))),
        np.asarray([0.5 * (init_[-1] + init_[-2])])
    ))

    return _init_ - time_ * (ord1_acc2(init_*flux_, vars_) - sink_)


@jit(nopython=True)
def lx_loop(time, init_, vars_, _flux_, _sink_, stability):
    """ loop for lx scheme """

    time_ = time_step_util(vars_, _flux_(init_, vars_), stability)

    tidx = np.arange(time[0], time[-1] + time_, time_)

    pts = np.asarray((tidx.size+1., np.asarray(time).size+100., 100.)).min()//1
    sols = np.asarray(init_).reshape(1, -1)
    tims = np.asarray(tidx[0]).reshape(1, -1)

    _yvar = sols[0]

    for itrs in range(tidx[:-1].size):

        next_ = lx_next(
            _yvar, vars_,
            _flux_(_yvar, vars_), _sink_(_yvar, vars_),
            tidx[itrs + 1] - tidx[itrs])

        if ((itrs + 1) % (tidx.size//pts)) == 0:
            sols = np.concatenate(
                (sols, np.asarray(next_).reshape(1, -1)), axis=0)
            tims = np.concatenate(
                (tims, np.asarray(tidx[itrs]).reshape(1, -1)), axis=0)

        _yvar = next_

    return tims, sols
