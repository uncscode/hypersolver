""" utilities for hypersolver """

import numpy as np


def term_util(term, orig):
    """ regularize term """
    if isinstance(term, np.ndarray) and np.array(term).shape == orig.shape:
        return term
    return np.full_like(orig, term, dtype=np.float)
