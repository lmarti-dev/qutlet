from typing import Tuple, Dict

import pytest
import numpy as np

import qutip

from fauvqe import TraceDistance, AbstractModel


class MockModel(AbstractModel):
    def __init__(self):
        super().__init__("GridQubit", [1, 1])

    def to_json_dict(self) -> Dict:
        return {}

    @classmethod
    def from_json_dict(cls, params: Dict):
        return cls()

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0]), np.array([0])
    
    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()
    
    def copy():
        return MockModel()

@pytest.mark.parametrize(
    "state1, state2, res",
    [
        (np.array([1,0,0,0]), np.array([0,1,0,0,]), 1),
        (1/np.sqrt(2) *np.array([1,1,0,0]), 1/np.sqrt(2) *np.array([1,1,0,0,]), 0)
    ],
)
def test_pure(state1, state2, res):
    model = MockModel()
    
    objective = TraceDistance(model, state1)
    
    dist = objective.evaluate(state2)
    print(dist)
    assert abs(dist - res) < 1e-10

@pytest.mark.parametrize(
    "state1, state2, res",
    [
        (qutip.Qobj(0.5*np.array([[1, 1], [1, 1]])), qutip.Qobj(0.5*np.array([[1, -1], [-1, 1]])), 1),
        (qutip.Qobj(0.5*np.array([[1, 0], [0, 1]])), qutip.Qobj(0.5*np.array([[1, 1], [1, 1]])), 0.5)
    ],
)
def test_mixed(state1, state2, res):
    model = MockModel()
    
    objective = TraceDistance(model, state1)
    
    dist = objective.evaluate(state2)
    print(state1, dist)
    assert abs(dist - res) < 1e-10

@pytest.mark.parametrize(
    "state",
    [
        (qutip.Qobj(0.5*np.array([[1, -1], [-1, 1]])))
    ],
)
def test_json(state):
    model = MockModel()
    
    objective = TraceDistance(model, state)
    print(objective)
    
    json = objective.to_json_dict()
    
    objective2 = TraceDistance.from_json_dict(json)
    
    assert (objective == objective2)

def test_exception():
    model = MockModel()
    test = TraceDistance(model, np.zeros(2))
    with pytest.raises(AssertionError):
        assert test.evaluate("Foo")