"""
    Test class ANNNI(SpinModelFC)
"""
# external import
import cirq
import pytest
import numpy as np


# internal import
from fauvqe import ANNNI, Converter, Ising


@pytest.mark.parametrize(
    "n, J, k, h, boundaries, hamiltonian",
    [
        (
            2,
            1,
            1,
            1,
            0,
            Ising(  "GridQubit",
                        [2,1],
                        1*np.ones((2-1,1)),
                        1*np.ones((2,1-1)),
                        1*np.ones((2,1)),
                        "X" )._hamiltonian
        ),
        (
            2,
            1,
            1,
            1,
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(i+1,0)) for i in range(2-1)]) +
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.GridQubit(i,0)) for i in range(2)]),
        ),
        (
            3,
            1,
            0.5,
            1,
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(i+1,0)) for i in range(3-1)]) +
            cirq.PauliSum.from_pauli_strings([-0.5*cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(i+2,0)) for i in range(3-2)]) +
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.GridQubit(i,0)) for i in range(3)]),
        ),
    ]
)
def test_constructor_hamiltonian(n, J, k, h, boundaries, hamiltonian):
    annni_obj = ANNNI(n, J, k, h, boundaries)

    assert annni_obj.qubittype == "GridQubit"
    assert len(annni_obj.n) == 2
    print(hamiltonian)
    print(annni_obj._hamiltonian)
    assert annni_obj._hamiltonian == hamiltonian

@pytest.mark.parametrize(
    "n, J, k, h, boundaries, expected_boundaries",
    [
        # Convert/use given boundary boundaries by shape of k
        (
            2,
            1,
            1,
            1,
            0,
            np.array([0,1]),
        ),
        (
            [1,3],
            1,
            1,
            1,
            0,
            np.array([1,0]),
        ),
        (
            [2,2],
            1,
            1,
            1,
            [1,1],
            np.array([1,1]),
        ),
        # Set boundaries by shape of J
        (
            3,
            np.ones((3)),
            1,
            1,
            None,
            np.array([0,1]),
        ),
        (
            [1,3],
            np.ones((2)),
            1,
            1,
            None,
            np.array([1,1]),
        ),
        (
            [1,3],
            np.ones((3)),
            1,
            1,
            None,
            np.array([1,0]),
        ),
        (
            [2,3],
            [np.ones((2,3)),np.ones((2,2))],
            1,
            1,
            None,
            np.array([0,1]),
        ),
        # Set boundaries by shape of k
        (
            [1,5],
            1,
            np.ones((3)),
            1,
            None,
            np.array([1,1]),
        ),
        (
            [1,5],
            1,
            np.ones((5)),
            1,
            None,
            np.array([1,0]),
        ),
    ]
)
def test_constructor_boundaries(n, J, k, h, boundaries, expected_boundaries):
    annni_obj = ANNNI(n, J, k, h, boundaries)
    assert (annni_obj.boundaries == expected_boundaries).all()

def test_converter():
    pass


#############################################
#                                           #
#               Test Asserts                #
#                                           #
#############################################
@pytest.mark.parametrize(
    "n, J, k, h, boundaries",
    [
        (
            2,
            1,
            1,
            1,
            None,
        ),
    ]
)
def test_constructor_errors(n, J, k, h, boundaries):
    with pytest.raises(AssertionError):
        annni_obj = ANNNI(n, J, k, h, boundaries)