""" hyper---bolic partial differential equations---solver

    hypersolver revolves around solving hyperbolic
    partial differential equations (PDEs) of the form

    ∂n/∂t + ∂(fn)/∂x = ∂n/∂t + f ∂n/∂x + n ∂f/∂x = g

    where

    n is a property of interest,
    x is an independent variable of interest,
    f is speed n moves along x, and
    g lumps sources and sinks

    functionally, n(x; t), f(x), and g(n; x; t)

    note, fn is the flux across x.

    Usage:
    >>> from hypersolver import solver
    >>> solver(n0, x, t, f, g, **kwargs)
    >>> # kwargs include "method", "backend", etc.

    available methods:
        - "lax_friedrichs" (default)
        - "lax_wendroff" (still unstable, wip)
        - "method_of_characteristics" (experimental)

    available backends:
        - "numpy" (default)
        - "jax" (experimental)
"""

import os

from hypersolver.step_solver import solver_


__version__ = "0.0.5"

__hyper_solvers__ = [
    "lax_friedrichs",
    "lax_wendroff",
    "method_of_characteristics",
]


def solver(
    *args,
    method="lax_friedrichs",
    backend=os.environ.get("BACKEND", "numpy"),
    **kwargs
):
    """ wrapper function to select solvers """

    os.environ["BACKEND"] = backend

    if method not in __hyper_solvers__:
        raise ValueError("method not supported")
    return solver_(*args, method=method, **kwargs)
