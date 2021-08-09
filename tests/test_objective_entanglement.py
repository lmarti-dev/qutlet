from typing import Tuple, Dict

import pytest
import numpy as np

import qutip

from fauvqe import Fidelity, AbstractModel


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
        (np.array([1,0,0,0]), np.array([0,1,0,0,]), 0),
        (1/np.sqrt(2) *np.array([1,1,0,0]), 1/np.sqrt(2) *np.array([1,1,0,0,]), 1)
    ],
)
def test_evaluate_pure(state, indices, typ, alpha, res):
    model = MockModel()
    
    objective = Entanglement(model, True, typ, alpha)
    
    ent = objective.evaluate(state, indices)
    print(ent)
    
    assert abs(ent - res) < 1e-10

@pytest.mark.parametrize(
    "state1, state2, res",
    [
        (qutip.Qobj(0.5*np.array([[1, 1], [1, 1]])), qutip.Qobj(0.5*np.array([[1, -1], [-1, 1]])), 0),
        (qutip.Qobj(0.5*np.array([[1, 0], [0, 1]])), qutip.Qobj(0.5*np.array([[1, 1], [1, 1]])), 1/np.sqrt(2))
    ],
)
def test_mixed(state1, state2, res):
    model = MockModel()
    
    objective = Fidelity(model, state1, pure=False)
    
    fid = objective.evaluate(state2)
    print(state1, fid)
    assert abs(fid - res) < 1e-10