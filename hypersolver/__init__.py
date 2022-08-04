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
        - "numba" (numpy + numba; experimental; suboptimal)
        - "jax" (experimental; suboptimal with no performance gain)

    available `solver_type`s:
        - "unsplit" (default)
        - "split" (soon; not yet available)
"""

import os


__version__ = "0.0.7"

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
    method=os.environ.get("HS_METHOD", "lax_friedrichs"),
    backend=os.environ.get("HS_BACKEND", "numpy"),
    verbosity=os.environ.get("HS_VERBOSITY", "0"),
    solver_type=os.environ.get("HS_SOLVER_TYPE", "unsplit"),
    **kwargs
):
    """ wrapper function to select solvers """

    if method not in __hyper_methods__ or \
            solver_type not in __hyper_solver_types__:
        raise ValueError("method not supported")

    os.environ["HS_METHOD"] = str(method)
    os.environ["HS_BACKEND"] = str(backend)
    os.environ["HS_VERBOSITY"] = str(verbosity)
    os.environ["HS_SOLVER_TYPE"] = str(solver_type)

    if backend == "numba" or os.environ.get("NUMBA_DISABLE_JIT") == "0":
        os.environ["HS_BACKEND"] = "numpy"
        os.environ["NUMBA_DISABLE_JIT"] = "0"
    else:
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    # pylint: disable=import-outside-toplevel
    from hypersolver.pde_solver_unsplit import solver_ as solver_upde
    # from hypersolver.pde_solver_split import solver_ as solver_spde
    from hypersolver.ode_solver import solver_ as solver_ode

    if method.startswith("rk"):
        return solver_ode(*args, method=method, **kwargs)

    # if method.endswith("_split"):
    #     return solver_spde(*args, method=method, **kwargs)

    return solver_upde(
        *args,
        method=method,
        **kwargs
    )
