"""
Test parent class AbstractModel;
    -test whether correct error messages occur
    -test whether initialisation set obj.qubits correctly 
        based on some examples
    -try to check test coverage?
"""
# external import
import pytest
import numpy as np

# internal import
from fauvqe import AbstractModel


class MockAbstractModel(AbstractModel):
    def copy(self):
        return MockAbstractModel()
        
    def to_json_dict(self):
        return {}

    def from_json_dict(self, dct):
        return MockAbstractModel(**dct)

    def energy(self):
        return np.array([])

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()

# test_AbstractModel_expected_errors
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
def test_AbstractModel(qubittype, n):
    with pytest.raises(AssertionError):
        MockAbstractModel(qubittype, n)


# TypeError
def test_AbstractModel00():
    with pytest.raises(TypeError):
        MockAbstractModel("NamedQubit", 1)

def test_diagonalise_erros():
    with pytest.raises(AssertionError):
        MockAbstractModel("GridQubit", [1, 1]).diagonalise(solver="numpy.sparse")
