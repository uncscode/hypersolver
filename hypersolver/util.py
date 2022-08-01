""" utilities for hypersolver
"""

import numpy as np


def term_util(
    term,
    orig,
):
    """ regularize term

        this utility "regularizes" the input "term"
        by making it look like the "orig" input
    """
    if isinstance(term, np.ndarray) and np.array(term).shape == orig.shape:
        return term
    return np.full_like(orig, term, dtype=np.float_)


def time_step_util(
    vars_vals,
    flux_term,
    stability,
):
    """ default time_step

        this utility calculates the default time_step
    """
    return stability * (
        vars_vals[1:] - vars_vals[:1]
    ).min() / flux_term.max()


def prep_next_step(
    stability,
    vars_vals,
    flux_term,
    init_vals,
):
    """ prep routine

        this utility prepares for next_step
        by calculating tim_step and copying next_vals
    """
    if stability is None:
        stability = np.array([0.98], dtype=np.float32)
    time_step = time_step_util(vars_vals, flux_term, stability)
    next_vals = init_vals.copy()
    return (time_step, next_vals)
