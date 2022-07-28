""" hyper---bolic partial differential equations---solver

    hypersolver revolves around solving hyperbolic
    partial differential equations (PDEs) of the form

    ∂n/∂t + ∂(fn)/∂x = ∂n/∂t + f ∂n/∂x - n ∂f/∂x = g

    where

    n is a property of interest,
    x is an independent variable of interest,
    f is speed n moves along x, and
    g lumps sources and sinks.

    Functionally, n(x; t), f(x; n), and g(x; t; n).

    Note, fn is the flux across x.

    Usage: solver(n0, x, t, fn, g, **kwargs)
    - solver = hypersolver.select_solver(method)
    - solver = hypersolver.method.solver

    available method names:
        - "lax_friedrichs" (default) for the Lax-Friedrichs scheme

"""


__version__ = "0.0.0"

from hypersolver import lax_friedrichs


def select_solver(method="lax_friedrichs"):
    """ wrapper function to select solvers """

    if method == "lax_friedrichs":
        return lax_friedrichs.solver

    raise ValueError("method not supported")
