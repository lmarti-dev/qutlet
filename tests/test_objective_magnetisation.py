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
def test_evaluate(field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Magnetisation(ising, field, 0,0)

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    expval = objective.evaluate(wavefunction)
    print(wavefunction, expval)

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