import pytest
import numpy as np

from fauvqe import Magnetisation, Ising

@pytest.mark.parametrize(
    "field",
    [
        ("Z"),
        ("Y"),
        ("X")
    ],
)
def test_simulate(field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Magnetisation(ising, field, 0,0)

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

@pytest.mark.parametrize(
    "state, res, field",
    [
        (np.array([1, 0, 0, 0], dtype=np.complex128), 1, "Z"),
        (0.25*np.eye(4, dtype=np.complex128), 0, "Z"),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128), 0, "Y"),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128), 0, "X")
    ],
)
def test_evaluate(state, res, field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 5})
    objective = Magnetisation(ising, field, 0,0)
    
    expval = objective.evaluate(state)
    assert abs(expval - res) < 1e-10

def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Magnetisation(ising, "Z", 0,0)
    print(objective)
    print(objective.model)
    json = objective.to_json_dict()
    
    objective2 = Magnetisation.from_json_dict(json)
    
    assert (objective == objective2)

def test_exception():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    with pytest.raises(AssertionError):
        assert Magnetisation(ising, "Foo", 0,0)