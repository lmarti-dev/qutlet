import pytest
import numpy as np

from fauvqe import Correlation, Ising

@pytest.mark.parametrize(
    "field",
    [
        ("Z"),
        ("Y"),
        ("X")
    ],
)
def test_corr(field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 5})
    objective = Correlation(ising, field, [0,0],[0,1])

    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )
    expval = objective.evaluate(wavefunction, q_map={ising.qubits[0][k]: k for k in range(2)})
    print(wavefunction, expval)

def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Correlation(ising, "Z", [0,0],[0,1])
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = Correlation.from_json_dict(json)
    
    assert (objective == objective2)

def test_exception():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    with pytest.raises(AssertionError):
        assert not Correlation(ising, "Foo", [0,0],[0,1])