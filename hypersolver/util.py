""" utilities for hypersolver """

import os
import warnings
import functools

import numpy as np
# pytype: disable=import-error
# pylint: disable=import-error


def set_xnp(backend=os.environ.get("HS_BACKEND", "numpy")):
    """ wrapper to set numpy or jax.numpy """

    if backend == "jax":
        warnings.warn(
            f"no more {backend} support, reverting to numpy")
        return np

    return np


xnp = set_xnp()


def set_jxt(backend=os.environ.get("HS_BACKEND", "numpy")):
    """ fake numba as a global namespace """

    if backend == "numba":
        warnings.warn(
            "experimental numba support")
        from numba import jit  # pylint: disable=import-outside-toplevel
        return jit

    def wrap(nopython=True, parallel=True, **kwargs):
        """ fake jit """
        _, _, _ = nopython, parallel, kwargs

        def wrapper(func):

            @functools.wraps(func)
            def inner(*args, **kwargs):
                returning = func(*args, **kwargs)
                return returning

            return inner

        return wrapper

    return wrap


jxt = set_jxt()


@jxt(nopython=True)
def term_util(term, orig):
    """ regularize term

        utility to "regularize" the input "term"
        by making it look like the "orig" input
    """

    return xnp.broadcast_arrays(
        xnp.asarray(term, dtype=orig.dtype), orig)[0]


# @jxt(nopython=True)
def func_util(func, _vals, _vars, **kwargs):
    """ evaluate function if one """
    return func(_vals, _vars, **kwargs) if callable(func) else func


@jxt(nopython=True)
def time_step_util(vars_vals, flux_term, stability=0.98):
    """ utility to calculate the default time_step
    """

    return (xnp.asarray(stability) * xnp.asarray(
        vars_vals[1:] - vars_vals[:-1]
    ).min() / xnp.asarray(flux_term).max()).item()
