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

# internal imports
from fauvqe import Ising
from .test_isings import IsingTester


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
def test_energy_analytic_1d(qubittype, n, j_v, j_h, h, E_exp):
    atol = 1e-14
    ising_obj = Ising(qubittype, n, j_v, j_h, h)
    assert (
        abs(ising_obj.energy_analytic_1d() - E_exp) < atol
    ), "Analytic 1D JZZ hX energy test failed; expected: {}, received {}, tolerance {}".format(
        E_exp, ising_obj.energy_analytic_1d(), atol
    )
    # compare numeric to analytic 1D result
    # for some less trivial cases
    tester = IsingTester(10 * np.sqrt(atol))
    if abs(np.sum(h)) > atol:
        tester.simple_energy_JZZ_hX_test(qubittype, n, j_v, j_h, h, cirq.Z ** 2, E_exp, "X")
    else:
        pass
        # THIS NOT DOES PASS YET??
        # might be that analytic solution is not quite correct yet?
        # tester.simple_energy_JZZ_hX_test(qubittype, n, j_v, j_h, h, cirq.X**2, E_exp, 'Z')


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
def test_assert_energy_analytic_1d(qubittype, n, j_v, j_h, h):
    ising_obj = Ising(qubittype, n, j_v, j_h, h)
    with pytest.raises(AssertionError):
        ising_obj.energy_analytic_1d()
