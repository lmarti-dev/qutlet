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
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(i+2,0)) for i in range(3-2)]) +
            cirq.PauliSum.from_pauli_strings([-cirq.X(cirq.GridQubit(i,0)) for i in range(3)]),
        ),
        (
            5,
            1,
            0.5,
            0.8,
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(np.mod(i+1,5),0)) for i in range(5-1)]) +
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(np.mod(i+2,5),0)) for i in range(5-2)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(i,0)) for i in range(5)]),
        ),
        (
            [1,5],
            1,
            0.5,
            0.8,
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(0,i))*cirq.Z(cirq.GridQubit(0,np.mod(i+1,5))) for i in range(5-1)]) +
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(0,i))*cirq.Z(cirq.GridQubit(0,np.mod(i+2,5))) for i in range(5-2)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(0,i)) for i in range(5)]),
        ),
        (
            5,
            1,
            0.5,
            0.8,
            0,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(np.mod(i+1,5),0)) for i in range(5)]) +
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(i,0))*cirq.Z(cirq.GridQubit(np.mod(i+2,5),0)) for i in range(5)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(i,0)) for i in range(5)]),
        ),
        (
            [1,5],
            1,
            0.5,
            0.8,
            0,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(0,i))*cirq.Z(cirq.GridQubit(0,np.mod(i+1,5))) for i in range(5)]) +
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(0,i))*cirq.Z(cirq.GridQubit(0,np.mod(i+2,5))) for i in range(5)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(0,i)) for i in range(5)]),
        ),
        (
            [2,2],
            1,
            0.5,
            0.8*np.ones((2,2)),
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j+1,i)) for i in range(2) for j in range(2-1)]) +
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j,i+1))  for i in range(2-1) for j in range(2)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(j,i)) for i in range(2) for j in range(2)]),
        ),
        (
            [2,2],
            1,
            0.5,
            0.8*np.ones((1, 2,2)),
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j+1,i)) for i in range(2) for j in range(2-1)]) +
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j,i+1))  for i in range(2-1) for j in range(2)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(j,i)) for i in range(2) for j in range(2)]),
        ),
        (
            [2,3],
            1,
            0.5,
            0.8*np.ones((1, 2,3)),
            1,
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j+1,i)) for i in range(3) for j in range(2-1)]) +
            cirq.PauliSum.from_pauli_strings([-cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j,i+1))  for i in range(3-1) for j in range(2)]) +
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j,i+2))  for i in range(3-2) for j in range(2)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(j,i)) for i in range(3) for j in range(2)]),
        ),
    ]
)
def test_constructor_hamiltonian(n, J, k, h, boundaries, hamiltonian):
    annni_obj = ANNNI(n, J, k, h, boundaries)

    assert annni_obj.qubittype == "GridQubit"
    assert len(annni_obj.n) == 2
    print("hamiltonian:\n{}\n".format(hamiltonian))
    print("annni_obj._hamiltonian:\n{}\n".format(annni_obj._hamiltonian))
    print("hamiltonian-annni_obj._hamiltonian:\n{}\n".format(hamiltonian-annni_obj._hamiltonian))
    assert annni_obj._hamiltonian == hamiltonian

@pytest.mark.parametrize(
    "n, J, k, h, boundaries, hamiltonian",
    [
       (
            np.array([2,3]),
            0.7,
            0.5,
            0.8*np.ones((1, 2,3)),
            1,
            cirq.PauliSum.from_pauli_strings([-0.7*cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j+1,i)) for i in range(3) for j in range(2-1)]) +
            cirq.PauliSum.from_pauli_strings([-0.7*cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j,i+1))  for i in range(3-1) for j in range(2)]) +
            cirq.PauliSum.from_pauli_strings([+0.5*cirq.Z(cirq.GridQubit(j,i))*cirq.Z(cirq.GridQubit(j,i+2))  for i in range(3-2) for j in range(2)]) +
            cirq.PauliSum.from_pauli_strings([-0.8*cirq.X(cirq.GridQubit(j,i)) for i in range(3) for j in range(2)]),
        ),
    ]
)
def test_constructor_hamiltonian2(n, J, k, h, boundaries, hamiltonian):
    _J_array = np.zeros((1, *n, *n))
    for n0 in range(n[0]):
        for n1 in range(n[1]-1):
            _J_array[0,n0,n1+1,n0,n1] = J
            _J_array[0,n0,n1,n0,n1+1] = J

    for n0 in range(n[0]-1):
        for n1 in range(n[1]):
            _J_array[0,n0+1,n1,n0,n1] = J
            _J_array[0,n0,n1,n0+1,n1] = J

    _k_array = np.zeros((1, *n, *n))
    for n0 in range(n[0]):
        for n1 in range(n[1]-2):
            _k_array[0,n0,n1+2,n0,n1] = k
            _k_array[0,n0,n1,n0,n1+2] = k

    for n0 in range(n[0]-2):
        for n1 in range(n[1]):
            _k_array[0,n0+2,n1,n0,n1] = k
            _k_array[0,n0,n1,n0+2,n1] = k

    annni_obj = ANNNI(n, _J_array, _k_array, h, boundaries)

    assert annni_obj.qubittype == "GridQubit"
    assert len(annni_obj.n) == 2
    print("hamiltonian:\n{}\n".format(hamiltonian))
    print("annni_obj._hamiltonian:\n{}\n".format(annni_obj._hamiltonian))
    print("hamiltonian-annni_obj._hamiltonian:\n{}\n".format(hamiltonian-annni_obj._hamiltonian))
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
            np.ones((3)),
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
            np.ones((3)),
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
            5,
            1,
            np.ones((3)),
            1,
            None,
            np.array([1,1]),
        ),
        (
            [5,1],
            1,
            np.ones((5)),
            1,
            None,
            np.array([0,1]),
        ),
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
        (
            [2,5],
            1,
            [np.ones((0,5)),np.ones((2,5))],
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


@pytest.mark.parametrize(
    "annni_obj, ising_obj",
    [
        (
            ANNNI(3, 1, 0, 1,1),
            Ising(  "GridQubit",
                    [3,1],
                    np.ones((3-1,1)),
                    np.ones((3,1-1)),
                    np.ones((3,1)),
                    "X")

       ),
       (
            ANNNI(3, 0.3, 0, 0.47,1),
            Ising(  "GridQubit",
                    [3,1],
                    0.3*np.ones((3-1,1)),
                    0.3*np.ones((3,1-1)),
                    0.47*np.ones((3,1)),
                    "X")

       ),
        (
            ANNNI([1,3], 0.3, 0, 0.47,[1,0]),
            Ising(  "GridQubit",
                    [1,3],
                    0.3*np.ones((1-1,3)),
                    0.3*np.ones((1,3)),
                    0.47*np.ones((1,3)),
                    "X")

       ),
       (
            ANNNI([2,2], 0.3, 0, 0.47,[1,2]),
            Ising(  "GridQubit",
                    [2,2],
                    0.3*np.ones((2-1,2)),
                    0.3*np.ones((2,2-1)),
                    0.47*np.ones((2,2)),
                    "X")

       ),
       (
            ANNNI([2,3], 0.3, 0, 0.47,[1,0]),
            Ising(  "GridQubit",
                    [2,3],
                    0.3*np.ones((2-1,3)),
                    0.3*np.ones((2,3)),
                    0.47*np.ones((2,3)),
                    "X")

       ),
       (
            ANNNI([3,2], 0.3, 0, 0.47,[0,1]),
            Ising(  "GridQubit",
                    [3,2],
                    0.3*np.ones((3,2)),
                    0.3*np.ones((3,2-1)),
                    0.47*np.ones((3,2)),
                    "X")

       ),
    ]
)
def test_energy(annni_obj, ising_obj):
    #print("np.array(ising_obj.energy()):\n{}\n".format(np.array(ising_obj.energy())))
    #print("np.array(annni_obj.energy()):\n{}\n".format(np.array(annni_obj.energy())))
    #print("ising_obj._hamiltonian:\n{}\n".format(ising_obj._hamiltonian))
    #print("annni_obj._hamiltonian:\n{}\n".format(annni_obj._hamiltonian))
    #print("ising_obj._hamiltonian-annni_obj._hamiltonian:\n{}\n".format(ising_obj._hamiltonian-annni_obj._hamiltonian))
    np.testing.assert_allclose(np.array(annni_obj.energy()), np.array(ising_obj.energy()), atol=1e-15, rtol=1e-15)

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
        (
            [2,3],
            np.ones((1,2,3,2,3)),
            1,
            1,
            None,
        ),
    ]
)
def test_constructor_errors(n, J, k, h, boundaries):
    with pytest.raises(AssertionError):
        annni_obj = ANNNI(n, J, k, h, boundaries)