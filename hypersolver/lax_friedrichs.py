""" Lax-Friedrics finite-difference scheme """

from hypersolver.util import prep_next_step


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
        init_step:  n
        vars_vals:  x
        flux_term:  f
        sink_term:  g
        stability:  λ = Δt/Δx where |fλ| ≤ 1, ∀ x

        outputs
        -------
        next_vals:  n

        numerics
        --------
        n(j+1, i) = (
            0.5 * (n(j,i+1) + n(j+1,i-1)) -
            time_step / (x(i+1) - x(i-1)) * (
                n(j,i+1) * f(i+1) -
                n(j,i-1) * f(i-1)
            ) +
            g(j,i) * time_step

        time_step = (
            stability *
            (x(i+1) - x(i-1)).min() /
            (f(i)).max()
        )
    """

    (time_step, next_vals) = prep_next_step(
        stability, vars_vals, flux_term, init_vals)

    next_vals[1:-1] = (
        0.5 * (init_vals[2:] + init_vals[:-2]) -
        1.0 * time_step / (vars_vals[2:] - vars_vals[:-2]) * (
            init_vals[2:] * flux_term[2:] -
            init_vals[:-2] * flux_term[:-2]
        ) + sink_term[1:-1] * time_step
    )

    next_vals[0] = (
        0.5 * (init_vals[1] + init_vals[0]) -
        1.0 * time_step / (vars_vals[1] - vars_vals[0]) * (
            init_vals[1] * flux_term[1] -
            init_vals[0] * flux_term[0]
        ) + sink_term[0] * time_step
    )

    next_vals[-1] = (
        0.5 * (init_vals[-1] + init_vals[-2]) -
        1.0 * time_step / (vars_vals[-1] - vars_vals[-2]) * (
            init_vals[-1] * flux_term[-1] -
            init_vals[-2] * flux_term[-2]
        ) + sink_term[-1] * time_step
    )

    return next_vals
