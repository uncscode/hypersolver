""" hyper---bolic partial differential equations---solver """

__version__ = "0.0.0"

from . import lax_friedrichs


def select_solver(method="lax_friedrichs"):
    """ wrapper function to select solvers """

    if method == "lax_friedrichs":
        return lax_friedrichs.solver

    raise ValueError("method not supported")
