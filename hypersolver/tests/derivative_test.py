""" test: calculating derivatives via central differencing """

import os
import pytest

from hypersolver.util import xnp as np
from hypersolver.derivative import ord1_acc2, ord2_acc2

xvar = np.linspace(0, 10, 1000)
yvar = np.sin(xvar)
dydx = np.cos(xvar)
d2ydx2 = - np.sin(xvar)


def test_ord1_acc2():
    """ test: central differncing: order=1, accuracy=2 """

    assert ord1_acc2(
        yvar, xvar).shape == dydx.shape

    assert ord1_acc2(
        yvar, xvar) == pytest.approx(dydx, abs=1e-2)

    if os.environ.get("HS_BACKEND", "numpy") == "numpy":
        assert (ord1_acc2(
            yvar, xvar) - dydx).sum() == pytest.approx(0.0, abs=1e-2)
    else:
        assert (ord1_acc2(
            yvar, xvar) - dydx).sum() == pytest.approx(
                (dydx-dydx).sum(), abs=1e-2)


def test_ord2_acc2():
    """ test: central differncing: order=2, accuracy=2 """

    assert ord2_acc2(
        yvar, xvar).shape == d2ydx2.shape

    assert ord2_acc2(
        yvar, xvar) == pytest.approx(d2ydx2, abs=1e-1)

    if os.environ.get("HS_BACKEND", "numpy") == "numpy":
        assert (ord2_acc2(
            yvar, xvar) - d2ydx2).sum() == pytest.approx(0.0, abs=1e-1)
    else:
        assert (ord2_acc2(
            yvar, xvar) - d2ydx2).sum() == pytest.approx(
                (d2ydx2-d2ydx2).sum(), abs=1e-1)
