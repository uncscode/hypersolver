""" hyper---bolic partial differential equations---solver

    hypersolver revolves around solving hyperbolic
    partial differential equations (PDEs) of the form

    ∂n/∂t + ∂(fn)/∂x = ∂n/∂t + f ∂n/∂x + n ∂f/∂x = g

    where

    n is a property of interest,
    x is an independent variable of interest,
    f is speed n moves along x, and
    g lumps sources and sinks

    functionally, n(x; t), f(x), and g(n; x)

    note, fn is the flux across x.

    Usage:
    >>> from hypersolver import solver
    >>> solver(n0, x, t, f, g, **kwargs)
    >>> # kwargs include "method", "backend", etc.

    available `method`s:
    pde:
        - "lax_friedrichs" (default)
        - "lax_wendroff"
        - "method_of_characteristics" (experimental)
    ode:
        - "rk2"

    available `backend`s:
        - "numpy" (default)
        - "jax" (experimental)

    available `solver_type`s:
        - "unsplit" (default)
        - "split" (soon; not yet available)
"""

import os

from hypersolver.pde_solver_unsplit import solver_ as solver_upde
# from hypersolver.pde_solver_split import solver_ as solver_spde
from hypersolver.ode_solver import solver_ as solver_ode


__version__ = "0.0.6"

__hyper_methods__ = [
    "lax_friedrichs",
    "lax_wendroff",
    "method_of_characteristics",
    "rk2",
]

__hyper_solver_types__ = [
    "unsplit",
    "split",
]


def solver(
    *args,
    method="lax_friedrichs",
    backend=os.environ.get("HS_BACKEND", "numpy"),
    verbosity=os.environ.get("HS_VERBOSITY", "0"),
    solver_type="unsplit",
    **kwargs
):
    """ wrapper function to select solvers """

    os.environ["HS_BACKEND"] = str(backend)
    os.environ["HS_VERBOSITY"] = str(verbosity)

    if method not in __hyper_methods__ or \
            solver_type not in __hyper_solver_types__:
        raise ValueError("method not supported")

    if method.startswith("rk"):
        return solver_ode(*args, method=method, **kwargs)
    # if method.endswith("_split"):
    #     return solver_spde(*args, method=method, **kwargs)
    return solver_upde(
        *args,
        method=method,
        **kwargs
    )
