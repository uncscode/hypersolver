""" test: utilities for hypersolver """

import os
import pytest

from hypersolver.util import xnp as np
from hypersolver.util import set_xnp
from hypersolver.util import term_util
from hypersolver.util import time_step_util


def test_set_xnp():
    """ test: wrapper to set numpy or jax.numy """

    if os.environ.get("HS_BACKEND", "numpy") == "jax":
        assert set_xnp().__package__ == "jax.numpy"
    else:
        assert set_xnp().__package__ == "numpy"


def test_term_util():
    """ test: regularize term"""

    assert term_util(
        3, np.asarray([1, 2, 3])
    ).shape == np.array([3, 3, 3]).shape

    assert term_util(
        np.asarray([1., 2., 3.]), np.array([1, 2, 3])
    ).shape == np.array([1, 2, 3]).shape

    assert term_util(
        3, np.asarray([1., 2., 3.])
    ).min() == 3

    assert term_util(
        3., np.asarray([1, 2, 3])
    ).max() == 3


def test_time_step_util():
    """ test: utility to calculate the default time_step

        NOTE: adding pytest.approx(..., abs=1e-6) for jax's float32
    """

    assert time_step_util(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
    ) == pytest.approx(np.asarray([0.98 * (2 - 1) / 3]), abs=1e-6)

    assert time_step_util(
        np.array([1, 2, 3]),
        np.array([1, 2, 3]),
        np.array([0.88])
    ) == pytest.approx(np.asarray([0.88 * (2 - 1) / 3]), abs=1e-6)
