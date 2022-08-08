""" Lax-Friedrics finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.util import time_step_util
from hypersolver.derivative import ord1_acc2


@jit(parallel=True)
def lx_next(
    init_vals,
    vars_vals,
    flux_term,
    sink_term,
    time_step,
):
    """ next step according to Lax-Friedrics finite-difference scheme

        ∂n/∂t + ∂(fn)/∂x = g

        inputs
        ------
        init_vals:  n
        vars_vals:  x
        flux_term:  f
        sink_term:  g
        time_step:  Δt

        outputs
        -------
        next_vals:  n

        numerics
        --------
        n(j+1, i) = n(j, i) + Δt (g - Δ(fn)/Δx)(j, i)

        Δt ≤ λΔx/f ∀ x
        Δ(fn)/Δx is first-order derivative with accuracy of 2
        n(j, i) = (n(j, i-1) + n(j, i+1))/2
    """

    _init_vals = np.concatenate((
        np.asarray([0.5 * (init_vals[1] + init_vals[0])]),
        np.asarray((0.5 * (init_vals[2:] + init_vals[:-2]))),
        np.asarray([0.5 * (init_vals[-1] + init_vals[-2])])
    ))

    return _init_vals - time_step * (
        ord1_acc2(init_vals*flux_term, vars_vals) - sink_term)


@jit(nopython=True)
def lx_loop(time, init_vals, vars_vals, _flux_term, _sink_term, stability):
    """ loop over """

    time_step = time_step_util(
        vars_vals, _flux_term(init_vals, vars_vals), stability
    )

    tidx = np.arange(
        time[0], time[-1] + time_step, time_step
    )

    pts = np.asarray((tidx.size+1., np.asarray(time).size+100., 100.)).min()//1
    print(pts, tidx.size, np.asarray(time).size)
    sols = np.asarray(init_vals).reshape(1, -1)
    tims = np.asarray(tidx[0]).reshape(1, -1)
    _yvar = sols[0]

    for itrs in range(tidx[:-1].size):

        next_vals = lx_next(
            _yvar,
            vars_vals,
            _flux_term(_yvar, vars_vals),
            _sink_term(_yvar, vars_vals),
            tidx[itrs + 1] - tidx[itrs]
        )

        if ((itrs + 1) % (tidx.size//pts)) == 0:
            sols = np.concatenate(
                (sols,
                 np.asarray(next_vals).reshape(1, -1)), axis=0)
            tims = np.concatenate(
                (tims,
                 np.asarray(tidx[itrs]).reshape(1, -1)), axis=0)
        _yvar = next_vals

    return tims, sols
