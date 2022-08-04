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
            "experimental jax support is suboptimal with no performance gain")
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        return jnp

    return np


xnp = set_xnp()


def set_jit(backend=os.environ.get("HS_BACKEND", "numpy")):
    """ fake numba as a global namespace """

    if backend == "numba":
        warnings.warn(
            "experimental numba support")
        from numba import jit  # pylint: disable=import-outside-toplevel
        return jit

    def wrap(nopython=True, parallel=True):
        """ fake jit """
        _, _ = nopython, parallel

        def wrapper(func):

            @functools.wraps(func)
            def inner(*args, **kwargs):
                returning = func(*args, **kwargs)
                return returning

            return inner

        return wrapper

    return wrap


jxt = set_jit()


# @jxt
def term_util(term, orig):
    """ regularize term

        utility to "regularize" the input "term"
        by making it look like the "orig" input
    """

    if isinstance(term, type(orig)) and xnp.asarray(term).shape == orig.shape:
        return term

    return xnp.full_like(orig, term)


# @jxt(parallel=True)
def func_util(func, _vals, _vars, **kwargs):
    """ evaluate function if one """
    return func(_vals, _vars, **kwargs) if callable(func) else func


# @jxt
def time_step_util(vars_vals, flux_term, stability):
    """ utility to calculate the default time_step
    """

    if stability is None:
        stability = xnp.asarray([0.98])

    return xnp.array(stability) * xnp.array(
        vars_vals[1:] - vars_vals[:-1]
    ).min() / xnp.array(flux_term).max()
