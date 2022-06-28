"""
    This file tests DrivenModel(AbstractModel)
"""
import imp
import cirq
import numpy as np
import pytest

from fauvqe.models.drivenmodel import DrivenModel
from fauvqe.models.ising import Ising

@pytest.mark.parametrize(
    "models, drives, drive_names, t",
    [
        (
            [
                Ising(  "GridQubit",
                        [1,2],
                        2*(np.random.rand(1-1,2)- 0.5),
                        2*(np.random.rand(1,2-1)- 0.5),
                        0*(np.random.rand(1,2)- 0.5),
                        "Z" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*(np.random.rand(1-1,2)- 0.5),
                        0*(np.random.rand(1,2-1)- 0.5),
                        2*(np.random.rand(1,2)- 0.5),
                        "X" ),
            ],
            [
                lambda t:  1,
                lambda t: np.sin(t)
            ],
            [
                "f(t) = 1",
                "f(t) = sin(t)"
            ],
            0
        ),
    ],
)
def test_constructor(models, drives, drive_names, t):
    driven_model = DrivenModel(models, drives, t)
    for i in range(len(drives)): drives[i].__name = drive_names[i]

    assert driven_model.models == models
    assert driven_model.drives == drives


def test_constructor_asserts():
    pass