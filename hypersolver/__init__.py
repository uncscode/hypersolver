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

from hypersolver.util import jxt as jit
from hypersolver.lax_friedrichs import lx_loop
from hypersolver.lax_wendroff import lw_loop
from hypersolver.runge_kutta import rk_loop


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


def set_solver(
    method=os.environ.get("HS_METHOD", "lax_friedrichs"),
    backend=os.environ.get("HS_BACKEND", "numpy"),
    verbosity=os.environ.get("HS_VERBOSITY", "0"),
    solver_type=os.environ.get("HS_SOLVER_TYPE", "unsplit"),
):
    """ wrapper function to select solvers """

    if method not in __hyper_methods__ or \
            solver_type not in __hyper_solver_types__:
        raise ValueError("method not supported")

    os.environ["HS_METHOD"] = str(method)
    os.environ["HS_BACKEND"] = str(backend)
    os.environ["HS_VERBOSITY"] = str(verbosity)
    os.environ["HS_SOLVER_TYPE"] = str(solver_type)

    @jit(nopython=True)
    def _solver(*args):
        """ jit a loop """
        if method == "lax_friedrichs":
            return lx_loop(*args)
        if method == "lax_wendroff":
            return lw_loop(*args)
        return rk_loop(*args)

    return _solver
