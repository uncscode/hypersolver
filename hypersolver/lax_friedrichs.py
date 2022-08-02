""" Lax-Friedrics finite-difference scheme """

from hypersolver.util import xnp as np
from hypersolver.util import time_step_util
from hypersolver.derivative import ord1_acc2


def lx_init(init_vals):
    """ initialize the array

        pad the array with the prescribed first and last values
    """

    return np.pad(
        init_vals,
        (1, 1),
        mode="constant",
        constant_values=(
            0.5 * (init_vals[1] + init_vals[0]),
            0.5 * (init_vals[-1] + init_vals[-2])
        ),)


def lx_next(
    init_vals,
    vars_vals,
    flux_term,
    sink_term,
    stability=None,
):
    """ next step according to Lax-Friedrics finite-difference scheme

        ∂n/∂t + ∂(fn)/∂x = g

        inputs
        ------
        init_vals:  n
        vars_vals:  x
        flux_term:  f
        sink_term:  g
        stability:  λ = Δt/Δx where |fλ| ≤ 1, ∀ x

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

    time_step = time_step_util(vars_vals, flux_term, stability)

    _init_vals = lx_init(0.5 * (init_vals[2:] + init_vals[:-2]))

    return _init_vals - time_step * (
        ord1_acc2(init_vals*flux_term, vars_vals) - sink_term)
