import pytest
import numpy as np
import cirq

from fauvqe import Correlation, Ising

@pytest.mark.parametrize(
    "field",
    [
        ("Z"),
        ("Y"),
        ("X"),
        ("S")
    ],
)
def test_simulate(field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 5})
    if(field == "S"):
        field = cirq.PauliString(cirq.X(q) for q in ising.qubits[0])
    objective = Correlation(ising, field)
    
    wavefunction = objective.simulate(
        param_resolver=ising.get_param_resolver(ising.circuit_param_values)
    )

@pytest.mark.parametrize(
    "state, res, field",
    [
        (np.array([1, 0, 0, 0], dtype=np.complex128), 1, "Z"),
        (0.25*np.eye(4, dtype=np.complex128), 0, "Z"),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128), 1, "Y"),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128), 1, "X"),
        (1/np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128), 1, "S"),
        (0.25*np.array([[1, 1, 0, 0], 
                        [1, 1, 0, 0], 
                        [0, 0, 1, -1], 
                        [0, 0, -1, 1]], dtype=np.complex128), 0, "S")
    ],
)
def test_evaluate(state, res, field):
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_circuit("qaoa", {"p": 5})
    if(field == "S"):
        field = cirq.PauliString(cirq.X(q) for q in ising.qubits[0])
    objective = Correlation(ising, field)
    
    expval = objective.evaluate(state)
    assert abs(expval - res) < 1e-10 
    
def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Correlation(ising, "Z")
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = Correlation.from_json_dict(json)
    
    assert (objective == objective2)

def test_exception():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    with pytest.raises(AssertionError):
        assert not Correlation(ising, "Foo")