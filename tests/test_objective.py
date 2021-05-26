import pytest
import numpy as np

from fauvqe import Objective, Ising


def test_objective():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    with pytest.raises(TypeError):
        Objective(ising)
