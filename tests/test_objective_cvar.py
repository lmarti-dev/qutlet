import pytest
import numpy as np

from fauvqe import CVaR, Ising


def test_cvar():
    pass


def test_cvar_initialiser():
    pass


def test_calc_cvar():
    cvar = CVaR._calc_cvar(np.array([1]), np.array([0]), alpha=1)
    print(cvar)


def test_calc_cvar_error():
    pass


@pytest.mark.parametrize(
    "alpha, field",
    [
        (0.1, "Z"),
        (0.5, "X"),
        (0.1, "X"),
        (0.5, "Z"),
    ],
)
def test_cvar_repr(alpha, field):
    # ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    # cvar = CVaR(initialiser=ising, alpha=alpha, field=field)
    # assert repr(cvar) == "<cVaR field={} alpha={}>".format(field, alpha)
    pass
