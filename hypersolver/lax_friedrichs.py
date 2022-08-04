""" Lax-Friedrics finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.util import jxt as jit
from hypersolver.derivative import ord1_acc2


@jit(nopython=True, parallel=True)
def lx_init(init_vals):
    """ initialize the array

        pad the array with the prescribed first and last values
    """

    return np.concatenate((
        np.asarray([0.5 * (init_vals[1] + init_vals[0])]),
        init_vals,
        np.asarray([0.5 * (init_vals[-1] + init_vals[-2])])
    ))


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
