from typing import Tuple, Dict

import pytest
import numpy as np

import qutip

from fauvqe import Entanglement, AbstractModel


class MockModel(AbstractModel):
    def __init__(self, n: int):
        self._n = n
        super().__init__("GridQubit", [1, n])

    def to_json_dict(self) -> Dict:
        return {}

    @classmethod
    def from_json_dict(cls, params: Dict):
        return cls()

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0]), np.array([0])
    
    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()
    
    def copy(self):
        return MockModel(self._n)

@pytest.mark.parametrize(
    "state, indices, typ, alpha, res",
    [
        (0.5*np.array([1,1,1,1]), [0], 'Neumann', None, 0),
        (0.5*np.array([1,1,1,1]), [0], 'Renyi', 2, 0),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 'Neumann', None, np.log(2)),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 'Renyi', 0.5, np.log(2))
    ],
)
def test_evaluate_pure(state, indices, typ, alpha, res):
    model = MockModel(2)
    
    objective = Entanglement(model, True, typ, alpha, indices)
    
    ent = objective.evaluate(state)
    print(ent)
    
    assert abs(ent - res) < 1e-10

@pytest.mark.parametrize(
    "state, indices, typ, alpha, res",
    [
        (0.5*np.array([1,1,1,1]), [0], 'Neumann', None, 0.4164955306996875),
        (0.5*np.array([1,1,1,1]), [0], 'Renyi', 2, 0.2876820724517809),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 'Neumann', None, 0.5623351446188083),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 'Renyi', 0.5, 0.6238107163648713)
    ],
)
def test_evaluate_mixed(state, indices, typ, alpha, res):
    model = MockModel(2)
    
    objective = Entanglement(model, False, typ, alpha, indices)
    
    mat = 0.5 * np.kron(state.reshape(1, 4), state.reshape(4, 1)) + 0.5 * np.array([[0, 0, 0, 0],
                                                                       [0, 0, 0, 0],
                                                                       [0, 0, 1, 0],
                                                                       [0, 0, 0, 0]])
    
    
    q = qutip.Qobj(mat, dims=[[2 for k in range(objective._n)], [2 for k in range(objective._n)]])
    
    ent = objective.evaluate(q)
    print(ent)
    
    assert abs(ent - res) < 1e-10