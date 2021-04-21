"""
Test parent class Initialiser;
    -test whether correct error messages occur
    -test whether initialisation set obj.qubits correctly 
        based on some examples
    -try to check test coverage?
"""
# external import
import cirq
import pytest
import numpy as np

# internal import
from fauvqe import Initialiser

# test_Initialiser_expected_errors
# AssertionError
@pytest.mark.parametrize(
    "qubittype, n",
    [
        ("NameQubit", "a"),
        ("NamedQubit", ["a", 2]),
        (
            "LineQubit",
            -1,
        ),  # "Error in qubit initialisation: n needs to be natural Number for LineQubit, received: n = -1"),
        ("LineQubit", 0),
        ("LineQubit", 1.0),
        ("GridQubit", np.array([1, 1.0])),
        ("GridQubit", [0, 1]),
        ("GridQubit", [2, 1.0]),
        ("GridQubit", [1.0, 2]),
        # ('GridQubit', 2),
        ("GridQubit", [1, 2, 3]),
    ],
)
def test_Initialiser(qubittype, n):
    with pytest.raises(AssertionError):
        initialiser_obj = Initialiser(qubittype, n)


# TypeError
def test_Initialiser00():
    with pytest.raises(TypeError):
        initialiser_obj = (Initialiser("NamedQubit", 1),)
