#External imports
import numpy as np
import pytest
from typing import Tuple, Dict

#Internal imports
from fauvqe import Fidelity, Ising

@pytest.mark.parametrize(
    "state1, state2, res",
    [
        (
            np.array([1, 0, 0, 0]),
            np.array(
                [
                    0,
                    1,
                    0,
                    0,
                ]
            ),
            0,
        ),
        (
            1 / np.sqrt(2) * np.array([1, 1, 0, 0]),
            1
            / np.sqrt(2)
            * np.array(
                [
                    1,
                    1,
                    0,
                    0,
                ]
            ),
            1,
        ),
         (
            np.array([1, 1, 0, 0])/ np.sqrt(2) ,
            np.array([1, 1, 0, 0])/ np.sqrt(2) ,
            1,
        ),
        (
            np.array([1, 1, 0, 0])/ np.sqrt(2) ,
            np.array([0, 1, 1, 0])/ np.sqrt(2) ,
            (1/2)**2,
        ),
    ],
)
def test_evaluate_pure(state1, state2, res):
    model = Ising(  "GridQubit", 
                    [1, 2], 
                    np.ones((0, 2)), 
                    np.ones((1, 1)), 
                    np.ones((1, 2))
                    )

    objective = Fidelity(model, state1)

    fid = objective.evaluate(state2)
    print(fid)
    assert abs(fid - res) < 1e-10


@pytest.mark.parametrize(
    "state1, state2, ground_truth",
    [
        (
            0.5*np.array([[1, 1], [1, 1]]), 
            0.5*np.array([[1, -1], [-1, 1]]), 
            0
        ),
        (
            np.array([1, 0]), 
            0.5*np.array([[1, -1], [-1, 1]]), 
            0.5
        ),
        (
            0.5*np.array([[1, 0], [0, 1]]), 
            0.5*np.array([[1, 1], [1, 1]]), 
            0.5
        )
    ],
)
def test_evaluate_mixed(state1, state2, ground_truth):
    model = Ising(  "GridQubit", 
                    [1, 1], 
                    np.ones((0, 1)), 
                    np.ones((1, 0)), 
                    np.ones((1, 1))
                    )

    objective = Fidelity(model, state1)

    fid = objective.evaluate(state2)
    print(state1, fid)
    assert abs(fid - ground_truth) < 1e-10

def test_simulate():
    ising = Ising(
        "GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2))
    )
    ising.set_circuit("qaoa", {"p": 5})
    objective = Fidelity(ising, 0.25 * np.ones(4))

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

@pytest.mark.parametrize(
    "state",
    [(0.5 * np.array([[1, -1], [-1, 1]]))],
)
def test_json(state):
    model = Ising(
        "GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2))
    )
    model.set_circuit("qaoa", {"p": 5})

    objective = Fidelity(model, state)
    print(objective)
    json = objective.to_json_dict()

    objective2 = Fidelity.from_json_dict(json)

    assert objective == objective2

def test_exceptions():
    model = Ising(
        "GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2))
    )
    model.set_circuit("qaoa", {"p": 5})

    objective = Fidelity(model, np.zeros(2))
    with pytest.raises(AssertionError):
        assert objective.evaluate("Foo")
    
    with pytest.raises(AssertionError):
        assert objective.evaluate(np.array([0, 1]))
