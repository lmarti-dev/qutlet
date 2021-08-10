import pytest
import numpy as np

from fauvqe import Energy, Ising

@pytest.mark.parametrize(
    "field",
    [
        ("Z"),
        ("X")
    ],
)
def test_energy(field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Energy(ising, field)

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    expval = objective.evaluate(wavefunction)
    print(wavefunction, expval)

def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Energy(ising, field="Z")
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = Energy.from_json_dict(json)
    
    assert (objective == objective2)