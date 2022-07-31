""" hyper---bolic partial differential equations---solver

    hypersolver revolves around solving hyperbolic
    partial differential equations (PDEs) of the form

    ∂n/∂t + ∂(fn)/∂x = ∂n/∂t + f ∂n/∂x - n ∂f/∂x = g

    where

    n is a property of interest,
    x is an independent variable of interest,
    f is speed n moves along x, and
    g lumps sources and sinks

    functionally, n(x; t), f(x), and g(x; t; n)

    note, fn is the flux across x.

    Usage:
    >>> from hypersolver import select_solver
    >>> solver = select_solver(method)
    >>> solver(n0, x, t, f, g, **kwargs)

    available methods:
        - "lax_friedrichs" (default)
        - "lax_wendroff"

"""


__version__ = "0.0.4"

__hyper_solvers__ = [
    "lax_friedrichs",
    "lax_wendroff",
    "method_of_characteristics",
]

from hypersolver.basic_solver import solver


def select_solver(method="lax_friedrichs"):
    """ wrapper function to select solvers """
    if method not in __hyper_solvers__:
        raise ValueError("method not supported")
    return solver(method)
