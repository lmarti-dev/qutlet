
# external imports
import cirq
import numpy as np
import pytest
from typing import Dict, Tuple

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

def test_plot_heatmap():
    n = [2,2]
    mock_model = MockModel(n)

    # Dummy to generate 'empty circuit'
    for i in range(mock_model.n[0]):
        for j in range(mock_model.n[1]):
            mock_model.circuit.append(cirq.Z(mock_model.qubits[i][j]) ** 2)

    test_state = mock_model.simulator.simulate(mock_model.circuit).state_vector()
    test_state = test_state / np.linalg.norm(test_state)

    plot_heatmap(mock_model, test_state)

@pytest.mark.parametrize(
    "n, test_gate, value_map_ground_truth, apply_to",
    [
        #############################################################
        #                   1 qubit tests                           #
        #############################################################
        (
            [1, 1],
            cirq.Z,
            {(0, 0): -1.0},
            [],
        ),
        #############################################################
        #                   2 qubit tests                           #
        #############################################################
        (
            [1, 2],
            cirq.Z,
            {(0, 0): -1.0, (0, 1): -1.0},
            [],
        ),
        (
            [2, 1],
            cirq.Z,
            {(0, 0): -1.0, (1, 0): -1.0},
            [],
        ),
        #############################################################
        #                   4 qubit tests                           #
        #############################################################
        # Okay this Z is just a phase gate:
        (
            [2, 2],
            cirq.Z,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0},
            [],
        ),
        (
            [2, 2],
            cirq.Z ** 2,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0},
            [],
        ),
        # X is spin flip |0000> -> |1111>:
        (
            [2, 2],
            cirq.X,
            {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0},
            [],
        ),
        # H : |0000> -> 1/\sqrt(2)**(n/2) \sum_i=0^2**1-1 |i>
        (
            [2, 2],
            cirq.H,
            {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0},
            [],
        ),
        # Test whether numbering is correct
        (
            [2, 2],
            cirq.X,
            {(0, 0): 1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): -1.0},
            np.array([[1, 0], [0, 0]]),
        ),
        (
            [2, 2],
            cirq.X,
            {(0, 0): -1.0, (0, 1): 1.0, (1, 0): -1.0, (1, 1): -1.0},
            np.array([[0, 1], [0, 0]]),
        ),
        (
            [2, 2],
            cirq.X,
            {(0, 0): -1.0, (0, 1): -1.0, (1, 0): 1.0, (1, 1): -1.0},
            np.array([[0, 0], [1, 0]]),
        ),
        (
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
        "Value map test failed: value_map_ground_truth expected to be dictionary, received: {}".\
            format(type(value_map_ground_truth))

    atol = 1e-14
    if np.size(apply_to) == 0:
        apply_to = np.ones(n)

    # Create mock_model
    mock_model = MockModel(n)

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
    test_state = test_state / np.linalg.norm(test_state)

    # Test where calculated spin value map fits to expectation spin
    # value map dictionary vm_exp
    value_map = get_value_map_from_state(mock_model, test_state)
    assert len(value_map) == len(
        value_map_ground_truth
    ), "Value map test failed: length expected: {}, received: {}".format(
        len(value_map_ground_truth), len(value_map)
    )

    # If elements in value_map and value_map_ground_truth do not match receive KeyError and test fails
    for element in value_map:
        assert np.allclose(value_map[element], value_map_ground_truth[element], rtol=0, atol=atol),\
            "Value map test failed: for element {}, expected: {}, received: {}"\
                .format(element, value_map_ground_truth[element], value_map[element])

    