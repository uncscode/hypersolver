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
    >>> from hypersolver import set_solver
    >>> solver = set_solver(method="lax_friedrichs", backend="numpy")
    >>> solver(n0, x, t, f, g)

    available `method`s:
    pde:
        - "lax_friedrichs" (default)
        - "lax_wendroff"
        - "method_of_characteristics" (experimental)
    ode:
        - "runge_kutta_2"

    available `backend`s:
        - "numpy" (default)
        - "numba" (numpy + numba; experimental)

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
    "runge_kutta_2",
]


def set_solver(
    method=os.environ.get("HS_METHOD", "lax_friedrichs"),
    backend=os.environ.get("HS_BACKEND", "numpy"),
):
    """ wrapper function to select solvers """

    if method not in __hyper_methods__:
        raise ValueError("method not supported")

    os.environ["HS_METHOD"] = str(method)
    os.environ["HS_BACKEND"] = str(backend)

    @jit(nopython=True)
    def _solver(*args):
        """ jit a loop """
        if method == "lax_friedrichs":
            return lx_loop(*args)

        return lw_loop(*args) if method == "lax_wendroff" else rk_loop(*args)

    return _solver


solver = set_solver()
