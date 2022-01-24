"""
    Test to be added:
        -test periodic boundaries
        -1D TFIM analytic result is inconsistent with QAOA energy:
            E_QAOA < E_analytic seems clearly wrong.
        -Need better TFIM tests as <Z>_X = <X>_Z = 0; 
            hence using Z/X eigenstates is generally a bad idea
"""
# external imports
import pytest
import numpy as np
import cirq
from scipy.linalg import expm
import sympy

# internal imports
from fauvqe import Ising
from tests.test_isings import IsingTester

def test__eq__():
    n = [1,3]; boundaries = [1, 0]
    ising = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])),"Z")
    ising.set_circuit("qaoa")
    
    ising2 = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])),"Z")
    ising2.set_circuit("qaoa")

    #print("ising == ising2: \t {}".format(ising == ising2))
    assert (ising == ising2)

    ising.set_Ut()
    assert ising != ising2 


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, basis",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            "Z",
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            "X",
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            "X",
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            "X",
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            "X",
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            "X",
        ),
    ],
)
def test_copy(qubittype, n, j_v, j_h, h, basis):
    ising = Ising(qubittype, n, j_v, j_h, h, basis)
    ising.set_circuit("qaoa")
    ising2 = ising.copy()

    #Test whether the objects are the same
    assert( ising == ising2 )

    #But there ID is different
    assert( ising is not ising2 )


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, test_gate, E_exp, basis",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect -j_v/2 = -0.1:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 1)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X,
            -0.1,
            "Z",
        ),
        # Normalised energy; hence expect -j_v/2 = -0.1:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 1)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X ** 2,
            -0.1,
            "Z",
        ),
        # Normalised energy; hence expect -j_h/2 = -0.25
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X,
            -0.25,
            "Z",
        ),
        # Normalised energy; hence expect -j_h/2 = -0.25
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X ** 2,
            -0.25,
            "Z",
        ),
        # Normalised energy; hence expect - 2h/2= -h:
        (
            "GridQubit",
            [1, 2],
            np.zeros((0, 2)) / 2,
            np.zeros((1, 1)) / 5,
            np.ones((1, 2)) / 3,
            cirq.X ** 2,
            -1 / 3,
            "X",
        ),
        # Normalised energy; hence expect + 2h/2 = h
        (
            "GridQubit",
            [1, 2],
            np.zeros((0, 2)) / 2,
            np.zeros((1, 1)) / 5,
            np.ones((1, 2)) / 3,
            cirq.Z,
            1 / 3,
            "X",
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect -h
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.ones((2, 2)) / 3,
            cirq.X,
            -1 / 3,
            "X",
        ),
        # Normalised energy; hence expect -h
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.ones((2, 2)) / 3,
            cirq.X ** 2,
            -1 / 3,
            "X",
        ),
        # Normalised energy; hence expect h
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.ones((2, 2)) / 3,
            cirq.Z,
            1 / 3,
            "X",
        ),
        #############################################################
        #                                                           #
        #          Need to come up with XX, Z mixed cases           #
        #       Issue: For X/Z eigenstates the Z/X exp is 0         #
        #                                                           #
        #############################################################
        # Normalised energy; hence expect (-j_h-2h)/2 = -0.35
        # ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
        #    np.ones((2,1))/10, cirq.X, -0.35,  'X'),
        # Normalised energy; hence expect (-j_h+2h)/2 = -0.15
        # ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
        #    np.ones((2,1))/10, cirq.X**2, -0.15,  'X'),
    ],
)
def test_energy_JZZ_hX(qubittype, n, j_v, j_h, h, test_gate, E_exp, basis):
    # set numerical tolerance
    # Note here lower than in jZZ_hZ case due to much larger numeric error
    atol = 1e-7
    tester = IsingTester(atol)
    tester.simple_energy_JZZ_hX_test(qubittype, n, j_v, j_h, h, test_gate, E_exp, basis)


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, test_gate, E_exp, basis",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect -j_v = -0.2:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X,
            -0.2,
            "Z",
        ),
        # Normalised energy; hence expect -j_v = -0.2:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X ** 2,
            -0.2,
            "Z",
        ),
        # Normalised energy; hence expect -j_h = -0.5
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X,
            -0.5,
            "Z",
        ),
        # Normalised energy; hence expect -j_h = -0.5
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X ** 2,
            -0.5,
            "Z",
        ),
        # Normalised energy; hence expect - 2h/2= -h:
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            cirq.X ** 2,
            -1 / 3,
            "X",
        ),
        # Normalised energy; hence expect + 2h/2 = h
        (
            "GridQubit",
            [1, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((1, 2)) / 5,
            np.ones((1, 2)) / 3,
            cirq.Z,
            1 / 3,
            "X",
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect -h
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            cirq.X,
            -1 / 3,
            "X",
        ),
        # Normalised energy; hence expect -h
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            cirq.X ** 2,
            -1 / 3,
            "X",
        ),
        # Normalised energy; hence expect h
        (
            "GridQubit",
            [2, 2],
            np.zeros((2, 2)) / 2,
            np.zeros((2, 2)) / 5,
            np.ones((2, 2)) / 3,
            cirq.Z,
            1 / 3,
            "X",
        ),
        # Normalised energy; hence expect -h-j_v+j_h
        # ('GridQubit', [2, 2], np.ones((2,2))/2, -np.ones((2,2))/5, \
        #    np.ones((2,2))/3, cirq.Z, -1/3,  'X'),
        #############################################################
        #                                                           #
        #          Need to come up with XX, Z mixed cases           #
        #       Issue: For X/Z eigenstates the Z/X exp is 0         #
        #                                                           #
        #############################################################
        # Normalised energy; hence expect (-j_h-2h)/2 = -0.35
        # ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
        #    np.ones((2,1))/10, cirq.X, -0.35,  'X'),
        # Normalised energy; hence expect (-j_h+2h)/2 = -0.15
        # ('GridQubit', [2, 1], np.ones((1,1))/2, np.ones((2,0))/5, \
        #    np.ones((2,1))/10, cirq.X**2, -0.15,  'X'),
    ],
)
def test_energy_JZZ_hX_p_boundary(qubittype, n, j_v, j_h, h, test_gate, E_exp, basis):
    # set numerical tolerance
    # Note here lower than in jZZ_hZ case due to much larger numeric error
    atol = 1e-7
    tester = IsingTester(atol)
    tester.simple_energy_JZZ_hX_test(qubittype, n, j_v, j_h, h, test_gate, E_exp, basis)


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, test_gate, E_exp",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect -j_v/2 = -0.1:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 1)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X,
            -0.1,
        ),
        # Normalised energy; hence expect -j_v/2 = -0.1:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 1)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X ** 2,
            -0.1,
        ),
        # Normalised energy; hence expect -j_h/2 = -0.25
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X,
            -0.25,
        ),
        # Normalised energy; hence expect -j_h/2 = -0.25
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X ** 2,
            -0.25,
        ),
        # Normalised energy; hence expect -j_v/2 - 2h/2= -0.2:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 1)) / 5,
            np.ones((1, 2)) / 10,
            cirq.X,
            -0.2,
        ),
        # Normalised energy; hence expect -j_v/2 + 2h/2 = 0
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 1)) / 5,
            np.ones((1, 2)) / 10,
            cirq.X ** 2,
            0,
        ),
        # Normalised energy; hence expect (-j_h-2h)/2 = -0.35
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 10,
            cirq.X,
            -0.35,
        ),
        # Normalised energy; hence expect (-j_h+2h)/2 = -0.15
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 10,
            cirq.X ** 2,
            -0.15,
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect (-2j_h-2j_v-4h)/4 = -0.45
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 1)) / 5,
            np.ones((2, 2)) / 10,
            cirq.X,
            -0.45,
        ),
        # Normalised energy; hence expect (-2j_h-2j_v-4h)/4 = -0.45
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 1)) / 5,
            np.ones((2, 2)) / 10,
            cirq.X ** 2,
            -0.25,
        ),
    ],
)
def test_energy_JZZ_hZ(qubittype, n, j_v, j_h, h, test_gate, E_exp):
    # set numerical tolerance
    atol = 1e-14
    tester = IsingTester(atol)
    tester.simple_energy_JZZ_hZ_test(qubittype, n, j_v, j_h, h, test_gate, E_exp)


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, test_gate, E_exp",
    [
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect -j_v = -0.2:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X,
            -0.2,
        ),
        # Normalised energy; hence expect -j_v = -0.2:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.X ** 2,
            -0.2,
        ),
        # Normalised energy; hence expect -j_h = -0.5
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X,
            -0.5,
        ),
        # Normalised energy; hence expect -j_h = -0.5
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.X ** 2,
            -0.5,
        ),
        # Normalised energy; hence expect -j_v - h= -0.3:
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((1, 2)) / 10,
            cirq.X,
            -0.3,
        ),
        # Normalised energy; hence expect -j_v + h = -0.1
        (
            "GridQubit",
            [1, 2],
            np.ones((0, 2)) / 2,
            np.ones((1, 2)) / 5,
            np.ones((1, 2)) / 10,
            cirq.X ** 2,
            -0.1,
        ),
        # Normalised energy; hence expect -j_h-h = -0.6
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 10,
            cirq.X,
            -0.6,
        ),
        # Normalised energy; hence expect (-j_h+h) = -0.4
        (
            "GridQubit",
            [2, 1],
            np.ones((2, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 1)) / 10,
            cirq.X ** 2,
            -0.4,
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Normalised energy; hence expect (-2j_h-2j_v-4h)/4 = -0.45
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 10,
            cirq.X,
            -0.8,
        ),
        # Normalised energy; hence expect (-2j_h-2j_v+4h)/4 = -0.6
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)) / 10,
            cirq.X ** 2,
            -0.6,
        ),
    ],
)
def test_energy_JZZ_hZ_p_boundary(qubittype, n, j_v, j_h, h, test_gate, E_exp):
    # set numerical tolerance
    atol = 1e-14
    tester = IsingTester(atol)
    tester.simple_energy_JZZ_hZ_test(qubittype, n, j_v, j_h, h, test_gate, E_exp)


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, test_gate, vm_exp, apply_to",
    [
        #############################################################
        #                   1 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 1],
            np.zeros((0, 1)) / 2,
            np.zeros((1, 0)) / 5,
            np.zeros((1, 1)) / 10,
            cirq.Z,
            {(0, 0): -1.0},
            [],
        ),
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            "GridQubit",
            [1, 2],
            np.zeros((0, 2)) / 2,
            np.zeros((1, 1)) / 5,
            np.zeros((1, 2)) / 10,
            cirq.Z,
            {(0, 0): -1.0, (0, 1): -1.0},
            [],
        ),
        (
            "GridQubit",
            [2, 1],
            np.zeros((1, 1)) / 2,
            np.zeros((2, 0)) / 5,
            np.zeros((2, 1)) / 10,
            cirq.Z,
            {(0, 0): -1.0, (1, 0): -1.0},
            [],
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Okay this Z is just a phase gate:
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.Z,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0},
            [],
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.Z ** 2,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0},
            [],
        ),
        # X is spin flip |0000> -> |1111>:
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.X,
            {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0},
            [],
        ),
        # H : |0000> -> 1/\sqrt(2)**(n/2) \sum_i=0^2**1-1 |i>
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.H,
            {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0},
            [],
        ),
        # Test whether numbering is correct
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.X,
            {(0, 0): 1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0},
            np.array([[1, 0], [0, 0]]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.X,
            {(0, 0): -1.0, (0, 1): 1.0, (1, 0): -1.0, (1, 1): -1.0},
            np.array([[0, 1], [0, 0]]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.X,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): 1.0, (1, 1): -1.0},
            np.array([[0, 0], [1, 0]]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.zeros((1, 2)) / 2,
            np.zeros((2, 1)) / 5,
            np.zeros((2, 2)) / 10,
            cirq.X,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): 1.0},
            np.array([[0, 0], [0, 1]]),
        ),
    ],
)
def test_get_spin_vm(qubittype, n, j_v, j_h, h, test_gate, vm_exp, apply_to):
    # set numerical tolerance
    atol = 1e-14
    tester = IsingTester(atol)
    tester.simple_spin_value_map_test(qubittype, n, j_v, j_h, h, test_gate, vm_exp, apply_to)


# Missing: test print_spin properly
def test_print_spin_dummy():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        np.zeros((1, 2)) / 2,
        np.zeros((2, 1)) / 5,
        np.zeros((2, 2)) / 10,
    )

    # Dummy to generate 'empty circuit'
    for i in range(ising_obj.n[0]):
        for j in range(ising_obj.n[1]):
            ising_obj.circuit.append(cirq.Z(ising_obj.qubits[i][j]) ** 2)

    wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()
    ising_obj.print_spin(wf)


# Test energy_analytic_1d
# test for even and odd n and in both arguments
# start with consistity test whether these are equal.
# Test trivial cases h = 0, j= 0
# LATER add some value testing
@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, E_exp",
    [
        (
            "GridQubit",
            [2, 1],
            np.zeros((1, 1)),
            np.zeros((2, 0)),
            3.14 * np.ones((2, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [10, 1],
            np.zeros((9, 1)),
            np.zeros((10, 0)),
            3.14 * np.ones((10, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [3, 1],
            np.zeros((2, 1)),
            np.zeros((3, 0)),
            3.14 * np.ones((3, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [11, 1],
            np.zeros((10, 1)),
            np.zeros((11, 0)),
            3.14 * np.ones((11, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 2],
            np.zeros((0, 2)),
            np.zeros((1, 1)),
            3.14 * np.ones((1, 2)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 10],
            np.zeros((0, 10)),
            np.zeros((1, 9)),
            3.14 * np.ones((1, 10)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 3],
            np.zeros((0, 3)),
            np.zeros((1, 2)),
            3.14 * np.ones((1, 3)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 11],
            np.zeros((0, 11)),
            np.zeros((1, 10)),
            3.14 * np.ones((1, 11)),
            -3.14,
        ),
        (
            "GridQubit",
            [2, 1],
            3.14 * np.ones((2, 1)),
            3.14 * np.ones((2, 0)),
            np.zeros((2, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [10, 1],
            3.14 * np.ones((10, 1)),
            3.14 * np.ones((10, 0)),
            np.zeros((10, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [3, 1],
            3.14 * np.ones((3, 1)),
            3.14 * np.ones((3, 0)),
            np.zeros((3, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [11, 1],
            3.14 * np.ones((11, 1)),
            3.14 * np.ones((11, 0)),
            np.zeros((11, 1)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 2],
            3.14 * np.ones((0, 2)),
            3.14 * np.ones((1, 1)),
            np.zeros((1, 2)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 10],
            3.14 * np.ones((0, 10)),
            3.14 * np.ones((1, 10)),
            np.zeros((1, 10)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 3],
            3.14 * np.ones((0, 3)),
            3.14 * np.ones((1, 3)),
            np.zeros((1, 3)),
            -3.14,
        ),
        (
            "GridQubit",
            [1, 11],
            3.14 * np.ones((0, 11)),
            3.14 * np.ones((1, 11)),
            np.zeros((1, 11)),
            -3.14,
        ),
    ],
)
def test_energy_pfeuty_sol(qubittype, n, j_v, j_h, h, E_exp):
    atol = 1e-14
    ising_obj = Ising(qubittype, n, j_v, j_h, h)
    assert (
        abs(ising_obj.energy_pfeuty_sol() - E_exp) < atol
    ), "Analytic 1D JZZ hX energy test failed; expected: {}, received {}, tolerance {}".format(
        E_exp, ising_obj.energy_pfeuty_sol(), atol
    )
    # compare numeric to analytic 1D result
    # for some less trivial cases
    tester = IsingTester(30 * np.sqrt(atol))
    if abs(np.sum(h)) > atol:
        tester.simple_energy_JZZ_hX_test(qubittype, n, j_v, j_h, h, cirq.Z ** 2, E_exp, "X")
    else:
        pass
        # THIS NOT DOES PASS YET??
        # might be that analytic solution is not quite correct yet?
        # tester.simple_energy_JZZ_hX_test(qubittype, n, j_v, j_h, h, cirq.X**2, E_exp, 'Z')

@pytest.mark.parametrize("n",range(2, 11))
def test_consistency_pfeuty_sol(n):
    """
        This consistency check with random J and h fails sometimes

        May indicate numerical precision problem somewhere!!
    """
    atol = 1e-7
    # Get solution via scipy sparse matrix solver
    J_zz = 2*np.random.random(1) - 1
    h_x  = 2*np.random.random(1) - 1
    ising_obj = Ising("GridQubit",
        [1, n],
        np.zeros((0, n)),
        J_zz  * np.ones((1, n)),
        h_x * np.ones((1, n)),
        "X")

    print("J_zz: \t {}, h_x \t {}".format(J_zz, h_x))

    ising_obj.diagonalise()
    assert (min(abs(ising_obj.energy_pfeuty_sol() - ising_obj.eig_val)) < atol),\
            "Pfeuty solution inconsistent with scipy sparse eig solver; Pfeuty: {}, scipy.sparse {}, tolerance {}".\
            format(ising_obj.energy_pfeuty_sol(), ising_obj.eig_val, atol)


@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, field, val_exp, vec_exp",
    [
        (
            "GridQubit",
            [2, 1],
            np.zeros((1, 1)),
            np.zeros((2, 0)),
            np.array([0.5, 1]).reshape((2,1)), 
            "X",
            [-0.75, -0.25],
            np.transpose([
                [-0.5, -0.5, -0.5, -0.5],
                [ 0.5,  0.5, -0.5, -0.5],
            ]),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((2, 2)),
            "X",
            [-1, -1],
            np.transpose([
                [1.+0.j , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0       ],
                [0      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.+0.j  ],
            ]),
        ),
        (
           "GridQubit",
            [2, 2],
            np.array([0.25, 0.75]).reshape((1,2)),
            np.array([0.5, 1]).reshape((2,1)),
            np.ones((2, 2)),
            "Z",
            [-1.625, -0.75],
            np.transpose([
                [1.+0.j , 0, 0, 0, 0, 0, 0, 0, 0     , 0, 0, 0, 0, 0, 0, 0],
                [0      , 0, 0, 0, 0, 0, 0, 0, 1.+0.j, 0, 0, 0, 0, 0, 0, 0],
            ]),
        ),
    ]
)

def test_diagonalise(qubittype, n, j_v, j_h, h, field, val_exp, vec_exp):
    # Create Ising object
    np_sol           =  Ising(qubittype, n, j_v, j_h, h, field)
    scipy_sol        =  Ising(qubittype, n, j_v, j_h, h, field)
    sparse_scipy_sol =  Ising(qubittype, n, j_v, j_h, h, field)

    #Calculate analytic results by different methods
    np_sol.diagonalise(solver = 'numpy')
    scipy_sol.diagonalise(solver = 'scipy')
    sparse_scipy_sol.diagonalise()

    # Test whether found eigenvalues are all close up to tolerance
    np.testing.assert_allclose(scipy_sol.eig_val    , sparse_scipy_sol.eig_val , rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np_sol.eig_val [0:2]  , sparse_scipy_sol.eig_val , rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(val_exp          , sparse_scipy_sol.eig_val , rtol=1e-14, atol=1e-14)

    # Test whether found eigenvectors are all close up to tolerance and global phase
    # Note that different eigen vectors can have a different global phase; hence we assert them one by one
    # Here we only check ground state and first excited state
    # Further issue: abitrary for degenerate
    for i in range(2):
        if np.abs(sparse_scipy_sol.eig_val[0] - sparse_scipy_sol.eig_val [1]) > 1e-14:
            #assert(sparse_scipy_sol.val[0] == sparse_scipy_sol.val[1] )
            cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(scipy_sol.eig_vec[:,i] , sparse_scipy_sol.eig_vec[:,i], rtol=1e-14, atol=1e-14)
        
        cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(np_sol.eig_vec[:,i]    , scipy_sol.eig_vec[:,i], rtol=1e-14, atol=1e-14)
        cirq.testing.lin_alg_utils.assert_allclose_up_to_global_phase(vec_exp[:,i]       , scipy_sol.eig_vec[:,i], rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h, field, glue_axis, sol_circuit, sol_circuit_param",
    [
        (
            "GridQubit",
            [3, 1],
            np.ones((3, 1)),
            np.ones((3, 0)),
            np.ones((3, 1)),
            "X",
            0,
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(1, 0)), cirq.H.on(cirq.GridQubit(2, 0)),
                        cirq.H.on(cirq.GridQubit(3, 0)), cirq.H.on(cirq.GridQubit(4, 0)), cirq.H.on(cirq.GridQubit(5, 0)),
                        (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 0)), (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(1, 0)),
                        (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(2, 0)), (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(3, 0)),
                        (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(4, 0)), (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(5, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(3, 0), cirq.GridQubit(4, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(2, 0), cirq.GridQubit(3, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(5, 0), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(4, 0), cirq.GridQubit(5, 0)),),
            [sympy.Symbol('b0_g0'),sympy.Symbol('g0_g0'),sympy.Symbol('b0_g1'),sympy.Symbol('g0_g1')]
        ),
        (
            "GridQubit",
            [1, 3],
            np.ones((0, 3)),
            np.ones((1, 3)),
            np.ones((1, 3)),
            "X",
            1,
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0, 0)), cirq.H.on(cirq.GridQubit(0, 1)), cirq.H.on(cirq.GridQubit(0, 2)),
                        cirq.H.on(cirq.GridQubit(0, 3)), cirq.H.on(cirq.GridQubit(0, 4)), cirq.H.on(cirq.GridQubit(0, 5)),
                        (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 0)), (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 1)),
                        (cirq.X**sympy.Symbol('b0_g0')).on(cirq.GridQubit(0, 2)), (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(0, 3)),
                        (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(0, 4)), (cirq.X**sympy.Symbol('b0_g1')).on(cirq.GridQubit(0, 5)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 4)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(0, 5), cirq.GridQubit(0, 0)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g0'))).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                        (cirq.ZZ**(1.0*sympy.Symbol('g0_g1'))).on(cirq.GridQubit(0, 4), cirq.GridQubit(0, 5)),),
            [sympy.Symbol('b0_g0'),sympy.Symbol('g0_g0'),sympy.Symbol('b0_g1'),sympy.Symbol('g0_g1')]
        ),
    ]
)
def test_glues_circuit(qubittype, n, j_v, j_h, h, field, glue_axis, sol_circuit, sol_circuit_param):
    ising = Ising(qubittype, n, j_v, j_h, h, field)
    ising.set_circuit("qaoa")
    #print(ising.circuit)
    
    ising.glue_circuit(axis=glue_axis)
    #print(ising.circuit)

    ising2 = Ising(qubittype, 
                    [(2-glue_axis)*n[0], (1+glue_axis)*n[1]], 
                    np.concatenate((j_v, j_v), axis=glue_axis),
                    np.concatenate((j_h, j_h), axis=glue_axis) , 
                    np.concatenate((h, h), axis=glue_axis) , 
                    field)
    ising2.circuit = sol_circuit
    ising2.circuit_param = sol_circuit_param
    ising2.circuit_param_values = np.array([0]*len(ising2.circuit_param))
    #print(sol_circuit)

    #print("ising.circuit == ising2.circuit: \t {}".format(ising.circuit == ising2.circuit))
    #print("ising.hamiltonian == ising2.hamiltonian: \t {}".format(ising.hamiltonian == ising2.hamiltonian))
    #print("ising.circuit_param_values: \t{}".format(ising.circuit_param_values))
    #print("ising2.circuit_param_values: \t{}".format(ising2.circuit_param_values))
    #assert(ising.circuit == ising2.circuit)
    assert(ising == ising2)
        
# This is potentially a higher effort test:
@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n",
    [
        (
            [2,2]
        ),
        (
            [1,12]
        ),
    ]
)
def test_set_Ut(n):
    boundaries = [1, 0]
    ising = Ising(  "GridQubit", 
                    n, 
                    np.zeros((n[0]-boundaries[0], n[1])), 
                    np.zeros((n[0], n[1]-boundaries[1])), 
                    np.ones((n[0], n[1])),
                    "X",
                    n[0]*n[1]*np.pi)
    ising.set_Ut()
    #print(6*ising.eig_val)
    #print(np.round(ising._Ut, decimals=5))
    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(np.identity(2**np.size(ising.qubits)), ising._Ut,rtol=1.1, atol=1e-7)


def hamiltonian(n):
    N = 2**n
    Z = np.array([[1, 0],
                  [0, -1]])
    X = np.array([[0, 1], 
                  [1, 0]])
    ham = np.zeros(shape=(N, N))
    for k in range(n):
        tmpx = 1
        for m in range(n):
            if(m == k):
                tmpx = np.kron(tmpx, X)
            else:
                tmpx = np.kron(tmpx, np.eye(2))
        ham = ham + tmpx
    for k in range(n-1):
        tmpzz = 1
        for m in range(n):
            if(m == k or m == k+1):
                tmpzz = np.kron(tmpzz, Z)
            else:
                tmpzz = np.kron(tmpzz, np.eye(2))
        ham = ham + tmpzz
    return -1*ham

@pytest.mark.higheffort
@pytest.mark.parametrize(
    "n, use_dense",
    [
        (
            [1,4], True
        ),
        (
            [1,12], False
        ),
        (
            [1,12], True
        ),
    ]
)
def test_Ut_correctness(n, use_dense):
    boundaries = [1, 1]
    t=0.1
    ising = Ising(  "GridQubit", 
                    n, 
                    np.ones((n[0]-boundaries[0], n[1])), 
                    np.ones((n[0], n[1]-boundaries[1])), 
                    np.ones((n[0], n[1])),
                    "X",
                    t)
    ising.set_Ut(use_dense)
    res = expm(-1j*t*hamiltonian(n[1]))
    cirq.testing .lin_alg_utils.assert_allclose_up_to_global_phase(res, ising._Ut,rtol=1.1, atol=1e-7)

#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################
@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((0, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((3, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 0)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 3)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((3, 1)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((0, 1)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 3)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 0)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 2],
            np.ones((2, 2)) / 2,
            np.ones((2, 2)) / 5,
            np.ones((2, 1)),
        ),
    ],
)
def test_assert_set_jh(qubittype, n, j_v, j_h, h):
    with pytest.raises(AssertionError):
        Ising(qubittype, n, j_v, j_h, h)


# Test energy_analytic_1d assertions
@pytest.mark.parametrize(
    "qubittype, n, j_v, j_h, h",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((1, 2)) / 2,
            np.ones((2, 1)) / 5,
            np.ones((2, 2)),
        ),
        (
            "GridQubit",
            [2, 1],
            np.ones((1, 1)) / 2,
            np.ones((2, 0)) / 5,
            np.reshape(np.array((0, -1)), (2, 1)),
        ),
        (
            "GridQubit",
            [3, 1],
            np.reshape(np.array((0, -1)), (2, 1)),
            np.ones((3, 0)),
            np.ones((3, 1)),
        ),
        (
            "GridQubit",
            [1, 3],
            np.ones((0, 3)),
            np.reshape(np.array((0, -1)), (1, 2)),
            np.ones((1, 3)),
        ),
    ],
)
def test_assert_energy_pfeuty_sol(qubittype, n, j_v, j_h, h):
    ising_obj = Ising(qubittype, n, j_v, j_h, h)
    with pytest.raises(AssertionError):
        ising_obj.energy_pfeuty_sol()

@pytest.mark.parametrize(
    "qubittype, n, j_z_v, j_z_h, h, field",
    [
        (
            "GridQubit",
            [2, 2],
            np.ones((0, 2)) / 2,
            np.ones((2, 2)) / 7,
            np.ones((2, 2)),
            "blub"
        )]
)
def test_assert_field(qubittype, n, j_z_v, j_z_h, h, field):
    with pytest.raises(AssertionError):
        Ising(qubittype, n, j_z_v, j_z_h, h, field)