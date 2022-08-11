""" Lax-Wendroff finite-difference scheme

    ∂n/∂t + ∂(fn)/∂x = g

    inputs
    ------
    init_:  n
    vars_:  x
    flux_:  f
    sink_:  (g, g)
    time_:  Δt

    outputs
    -------
    next_:  n

    numerics
    --------
    n(j+1) =
        n(j) +
        Δt (g - Δ(fn)/Δx)(j) +
        0.5 (Δt)^2 (
            - Δf/Δx (-Δ(fn)/Δx + g)
            - f(-Δ(fn)^2/Δx^2 + Δg/Δx)
            + Δg/Δt
        )(j)

    Δt ≤ λΔx/f ∀ x
    Δ(s)/Δx is first-order derivative with accuracy of 2
    Δ(s)^/Δx^2 is second-order derivative with accuracy of 2
    n(j, i) =? (n(j, i-1) + n(j, i+1))/2

"""

from hypersolver.util import jxt as jit
from hypersolver.util import xnp as np
from hypersolver.util import term_util, time_step_util
from hypersolver.derivative import ord1_acc2, ord2_acc2


@jit(nopython=True)
def lw_next(init_, vars_, flux_, sink_, time_):
    """ next step according to Lax-Friedrics finite-difference scheme """

    flux_ = term_util(flux_, init_)

    sink_ = (term_util(sink_[0], vars_), term_util(sink_[1], vars_))

    return init_ + time_ * (
        sink_[1] - ord1_acc2(init_*flux_, vars_)
    ) + 0.5 * time_**2 * (
        -1.0 * ord1_acc2(flux_, vars_)*(
            -1.0 * ord1_acc2(init_*flux_, vars_) +
            sink_[1]
        ) - flux_ * (
            -1.0*ord2_acc2(flux_, vars_) +
            ord1_acc2(sink_[1], vars_)
        ) + (sink_[1] - sink_[0])/time_
    )


@jit(nopython=True)
def lw_loop(time, init_, vars_, _flux_, _sink_, stability):
    """ loop for lw scheme """

    time_ = time_step_util(vars_, _flux_(init_, vars_), stability)

    tidx = np.arange(time[0], time[-1] + time_, time_)

    pts = np.asarray((tidx.size+1., np.asarray(time).size+100., 100.)).min()//1
    sols = np.asarray(init_).reshape(1, -1)
    tims = np.asarray(tidx[0]).reshape(1, -1)

    _sink1 = _sink_(init_, vars_).reshape(vars_.shape)
    _sink2 = _sink_(init_, vars_).reshape(vars_.shape)

    _yvar = sols[0]

    for itrs in range(tidx[:-1].size):

        next_ = lw_next(
            _yvar, vars_,
            _flux_(_yvar, vars_), (_sink1, _sink2),
            tidx[itrs + 1] - tidx[itrs])

        if ((itrs + 1) % (tidx.size//pts)) == 0:
            sols = np.concatenate(
                (sols, np.asarray(next_).reshape(1, -1)), axis=0)
            tims = np.concatenate(
                (tims, np.asarray(tidx[itrs]).reshape(1, -1)), axis=0)

        _yvar = next_
        _sink1 = _sink2
        _sink2 = _sink_(next_, _yvar).reshape(vars_.shape)

    return tims, sols
