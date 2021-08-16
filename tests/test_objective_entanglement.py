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
    "state, indices, alpha, res",
    [
        (0.5*np.array([1,1,1,1]), [0], 1, 0),
        (0.5*np.array([1,1,1,1]), [0], 2, 0),
        (0.5*np.array([1,1,1,1]), None, 2, 0),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 1, np.log(2)),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 0.5, np.log(2))
    ],
)
def test_evaluate_pure(state, indices, alpha, res):
    model = MockModel(2)
    
    objective = Entanglement(model, alpha, indices)
    
    ent = objective.evaluate(state)
    print(ent)
    
    assert abs(ent - res) < 1e-10

def test_neumann_renyi():
    state = np.random.rand(4)
    state = state / np.linalg.norm(state)
    model = MockModel(2)
    alpha = 1 + 1e-10
    renyi = Entanglement(model, alpha)
    neumann = Entanglement(model, 1)
    
    renyi_ent = renyi.evaluate(state)
    neumann_ent = neumann.evaluate(state)
    
    assert abs(renyi_ent - neumann_ent) < 1e-4

@pytest.mark.parametrize(
    "state, indices, alpha, res",
    [
        (0.5*np.array([1,1,1,1]), [0], 1, 0.4164955306996875),
        (0.5*np.array([1,1,1,1]), [0], 2, 0.2876820724517809),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 1, 0.5623351446188083),
        (1/np.sqrt(2)*np.array([1,0,0,1]), [0], 0.5, 0.6238107163648713)
    ],
)
def test_evaluate_mixed(state, indices, alpha, res):
    model = MockModel(2)
    
    objective = Entanglement(model, alpha, indices)
    
    mat = 0.5 * np.kron(state.reshape(1, 4), state.reshape(4, 1)) + 0.5 * np.array([[0, 0, 0, 0],
                                                                       [0, 0, 0, 0],
                                                                       [0, 0, 1, 0],
                                                                       [0, 0, 0, 0]])
    
    
    q = qutip.Qobj(mat, dims=[[2 for k in range(objective._n)], [2 for k in range(objective._n)]])
    
    ent = objective.evaluate(q)
    print(ent)
    
    assert abs(ent - res) < 1e-10

@pytest.mark.parametrize(
    "indices, alpha",
    [
        ([0], 1)
    ],
)
def test_json(alpha, indices):
    model = MockModel(2)
    
    objective = Entanglement(model, alpha, indices)
    
    json = objective.to_json_dict()
    print(objective)
    
    objective2 = Entanglement.from_json_dict(json)
    
    assert (objective == objective2)

def test_exceptions():
    model = MockModel(2)
    
    with pytest.raises(NotImplementedError):
        assert Entanglement(model, "Foo", 0, [0])

@pytest.mark.parametrize(
    "state",
    [
        ("Foo")
    ],
)
def test_exceptions(state):
    model = MockModel(2)
    objective = Entanglement(model, 2, [0])
    with pytest.raises(NotImplementedError):
        objective.evaluate(state)