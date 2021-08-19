from typing import Tuple, Dict

import pytest
import numpy as np

from qutip import Qobj

from fauvqe import Entanglement, Ising


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
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    
    objective = Entanglement(model, alpha, indices)
    
    ent = objective.evaluate(state)
    print(ent)
    
    assert abs(ent - res) < 1e-10

def test_evaluate_neumann_renyi():
    state = np.random.rand(4)
    state = state / np.linalg.norm(state)
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
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
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    objective = Entanglement(model, alpha, indices)
    
    mat = 0.5 * np.kron(state.reshape(1, 4), state.reshape(4, 1)) + 0.5 * np.array([[0, 0, 0, 0],
                                                                       [0, 0, 0, 0],
                                                                       [0, 0, 1, 0],
                                                                       [0, 0, 0, 0]])
    
    
    q = Qobj(mat, dims=[[2 for k in range(objective._n)], [2 for k in range(objective._n)]])
    
    ent = objective.evaluate(q)
    print(ent)
    
    assert abs(ent - res) < 1e-10

def test_simulate():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 5})
    objective = Entanglement(ising, 1)
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

@pytest.mark.parametrize(
    "indices, alpha",
    [
        ([0], 1)
    ],
)
def test_json(alpha, indices):
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    objective = Entanglement(model, alpha, indices)
    
    json = objective.to_json_dict()
    print(objective)
    
    objective2 = Entanglement.from_json_dict(json)
    
    assert (objective == objective2)

@pytest.mark.parametrize(
    "state",
    [
        ("Foo")
    ],
)
def test_evaluate_exceptions(state):
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    objective = Entanglement(model, 2, [0])
    with pytest.raises(AssertionError):
        objective.evaluate(state)