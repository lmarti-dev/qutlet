
# external imports
import pytest
import numpy as np
import cirq

#internal imports
from fauvqe.utilities.visual import (
    plot_heatmap,
    print_non_zero,
    get_value_map_from_state,
)
from fauvqe.models.abstractmodel import (
    AbstractModel,
)
from tests.test_models_isings import IsingTester

class MockModel(AbstractModel):
    def __init__(self, n):
        super().__init__("GridQubit", n)

    def copy(self):
        return MockModel()
    
    def to_json_dict(self) -> Dict:
        return {}

    @classmethod
    def from_json_dict(cls, params: Dict):
        return cls()

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0]), np.array([0])
    
    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()

def test_print_spin_dummy():
    ising_obj = Ising(
        "GridQubit",
        [2, 2],
        np.ones((1, 2)) / 2,
        np.ones((2, 1)) / 5,
        np.zeros((2, 2)) / 10,
    )

    # Dummy to generate 'empty circuit'
    for i in range(ising_obj.n[0]):
        for j in range(ising_obj.n[1]):
            ising_obj.circuit.append(cirq.Z(ising_obj.qubits[i][j]) ** 2)

    wf = ising_obj.simulator.simulate(ising_obj.circuit).state_vector()
    ising_obj.print_spin(wf)

@pytest.mark.parametrize(
    "n, test_gate, value_map_ground_truth, apply_to",
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
            cirq.X,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): 1.0},
            np.array([[0, 0], [0, 1]]),
        ),
    ],
)
def test_get_value_map_from_state(  n, 
                                    test_gate, 
                                    value_map_ground_truth, 
                                    apply_to):
    # Check wether wm_exp is a dictionary
    assert dict == type(value_map_ground_truth),\
        "Ising, value map test failed: value_map_ground_truth expected to be dictionary, received: {}".\
            format(type(value_map_ground_truth))

    atol = 1e-14
    if np.size(apply_to) == 0:
        apply_to = np.ones(n)

    # Create mock_model
    mock_model = Ising(qubittype, n, j_v, j_h, h)

    # Dummy to generate 'empty circuit'
    for i in range(mock_model.n[0]):
        for j in range(mock_model.n[1]):
            mock_model.circuit.append(cirq.Z(mock_model.qubits[i][j]) ** 2)

    # Apply test gate
    for i in range(mock_model.n[0]):
        for j in range(mock_model.n[1]):
            if apply_to[i][j] == 1:
                mock_model.circuit.append(test_gate(mock_model.qubits[i][j]))

    # Simulate
    test_state = mock_model.simulator.simulate(mock_model.circuit).state_vector()

    # Renormalise wavefunction as required by qsim:
    test_state = test_state / np.sqrt(abs(test_state.dot(np.conj(test_state))))

    # Test where calculated spin value map fits to expectation spin
    # value map dictionary vm_exp
    value_map = get_value_map_from_state(mock_model, test_state)
    assert len(value_map) == len(
        value_map_ground_truth
    ), "Ising, value map test failed: length expected: {}, received: {}".format(
        len(value_map_ground_truth), len(value_map)
    )

    # If elements in value_map and value_map_ground_truth do not match receive KeyError and test fails
    for element in value_map:
        assert np.allclose(value_map[element], value_map_ground_truth[element], rtol=0, atol=atol),\
            "Ising, value map test failed: for element {}, expected: {}, received: {}"\
                .format(element, value_map_ground_truth[element], value_map[element])

    