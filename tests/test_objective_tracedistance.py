from typing import Tuple, Dict

import pytest
import numpy as np

from qutip import Qobj

from fauvqe import TraceDistance, Ising


@pytest.mark.parametrize(
    "state1, state2, res",
    [
        (np.array([1,0,0,0]), np.array([0,1,0,0,]), 1),
        (1/np.sqrt(2) *np.array([1,1,0,0]), 1/np.sqrt(2) *np.array([1,1,0,0,]), 0)
    ],
)
def test_evaluate_pure(state1, state2, res):
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    
    objective = TraceDistance(model, state1)
    
    dist = objective.evaluate(state2)
    print(dist)
    assert abs(dist - res) < 1e-10

@pytest.mark.parametrize(
    "state1, state2, res",
    [
        (Qobj(0.5*np.array([[1, 1], [1, 1]])), Qobj(0.5*np.array([[1, -1], [-1, 1]])), 1),
        (Qobj(0.5*np.array([[1, 0], [0, 1]])), Qobj(0.5*np.array([[1, 1], [1, 1]])), 0.5)
    ],
)
def test_evaluate_mixed(state1, state2, res):
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    
    objective = TraceDistance(model, state1)
    
    dist = objective.evaluate(state2)
    print(state1, dist)
    assert abs(dist - res) < 1e-10

def test_simulate():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = TraceDistance(ising, np.array([1, 0, 0, 0]))
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

@pytest.mark.parametrize(
    "state",
    [
        (Qobj(0.5*np.array([[1, -1], [-1, 1]])))
    ],
)
def test_json(state):
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    
    objective = TraceDistance(model, state)
    print(objective)
    
    json = objective.to_json_dict()
    
    objective2 = TraceDistance.from_json_dict(json)
    
    assert (objective == objective2)

def test_exception():
    model = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    model.set_circuit("qaoa", {"p": 5})
    test = TraceDistance(model, np.zeros(2))
    with pytest.raises(AssertionError):
        assert test.evaluate("Foo")