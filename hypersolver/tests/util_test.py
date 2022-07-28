""" test: utilities for hypersolver """

import numpy as np

from hypersolver.util import term_util


def test_term_util():
    """ test: regularize term"""

    assert term_util(
        3, np.array([1, 2, 3])
    ).shape == np.array([3, 3, 3]).shape

    assert term_util(
        np.array([1, 2, 3]), np.array([1, 2, 3])
    ).shape == np.array([1, 2, 3]).shape

    assert term_util(
        3, np.array([1, 2, 3])
    ).min() == 3

    assert term_util(
        3, np.array([1, 2, 3])
    ).max() == 3
