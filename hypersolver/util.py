""" utilities for hypersolver """

import os

import numpy as np
# pytype: disable=import-error
if os.environ.get("HS_BACKEND", "numpy") == "jax":
    import jax.numpy as jnp  # pylint: disable=import-error


def set_xnp(backend=os.environ.get("HS_BACKEND", "numpy")):
    """ wrapper to set numpy or jax.numpy """

    return jnp if backend == "jax" else np


xnp = set_xnp()


def term_util(term, orig):
    """ regularize term

        utility to "regularize" the input "term"
        by making it look like the "orig" input
    """

    if isinstance(term, type(orig)) and xnp.array(term).shape == orig.shape:
        return term

    return xnp.full_like(orig, term)


def func_util(func, _vals, _vars, **kwargs):
    """ evaluate function if one """
    return func(_vals, _vars, **kwargs) if callable(func) else func


def time_step_util(vars_vals, flux_term, stability):
    """ utility to calculate the default time_step
    """

    if stability is None:
        stability = xnp.array([0.98])

    return xnp.array(stability) * xnp.array(
        vars_vals[1:] - vars_vals[:-1]
    ).min() / xnp.array(flux_term).max()
