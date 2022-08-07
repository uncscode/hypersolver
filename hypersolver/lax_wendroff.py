""" Lax-Wendroff finite-difference scheme """

from hypersolver.util import jxt as jit
from hypersolver.util import term_util
from hypersolver.derivative import ord1_acc2, ord2_acc2


@jit(nopython=True)
def lw_next(
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
        sink_term:  (g, g)
        stability:  λ = Δt/Δx where |fλ| ≤ 1, ∀ x

        outputs
        -------
        next_vals:  n

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

    flux_term = term_util(flux_term, init_vals)

    _sink1 = term_util(sink_term[0], vars_vals)
    _sink2 = term_util(sink_term[1], vars_vals)
    sink_term = (_sink1, _sink2)

    return init_vals + time_step * (
        sink_term[1] - ord1_acc2(init_vals*flux_term, vars_vals)
    ) + 0.5 * time_step**2 * (
        -1.0 * ord1_acc2(flux_term, vars_vals)*(
            -1.0 * ord1_acc2(init_vals*flux_term, vars_vals) +
            sink_term[1]
        ) - flux_term * (
            -1.0*ord2_acc2(flux_term, vars_vals) +
            ord1_acc2(sink_term[1], vars_vals)
        ) + (sink_term[1] - sink_term[0])/time_step
    )
