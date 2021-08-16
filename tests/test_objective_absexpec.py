import pytest
import numpy as np
import cirq

from fauvqe import AbstractExpectationValue, Ising

def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = AbstractExpectationValue(ising, cirq.PauliString(cirq.X(ising.qubits[0][0])))
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = AbstractExpectationValue.from_json_dict(json)
    
    assert (objective == objective2)

def test_exception():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    test = AbstractExpectationValue(ising, None)
    with pytest.raises(NotImplementedError):
        assert test.evaluate(np.zeros(2, dtype=np.complex64), q_map={ising.qubits[0][k]: k for k in range(2)})