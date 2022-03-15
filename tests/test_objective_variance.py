import cirq
import numpy as np
import pytest

from fauvqe import AbstractModel, Ising, Variance

class MockAbstractModel(AbstractModel):
    def copy(self):
        return MockAbstractModel()

    def energy(self):
        return np.array([])

    def from_json_dict(self, dct):
        return MockInitialiser()

    def to_json_dict(self):
        return {}

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()

@pytest.mark.parametrize(
    "state, observables, variances, n",
    [
        # 1 Qubit
        (
            np.array([1,0]),
            cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(1)]),
            0,
            1,
        ),
        (
            np.array([1,0]),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(i)) for i in range(1)]),
            1,
            1,
        ),
        (
            np.array([1,0]),
            cirq.PauliSum.from_pauli_strings([cirq.Y(cirq.LineQubit(i)) for i in range(1)]),
            1,
            1,
        ),
        (
            np.array([1,0]),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(0)), cirq.Z(cirq.LineQubit(0))]),
            1,
            1,
        ),
        # 1 Qubit & List of observables
        (
            np.array([1,0]),
            [cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(1)]),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(i)) for i in range(1)]),
            cirq.PauliSum.from_pauli_strings([cirq.Y(cirq.LineQubit(i)) for i in range(1)])],
            [0,1,1 ],
            1
        ),
        # 2 Qubits
        (
            np.array([0.5,0.5, 0.5, 0.5]),
            cirq.PauliSum.from_pauli_strings([cirq.Z(cirq.LineQubit(i)) for i in range(2)]),
            2,
            2,
        ),
        (
            np.array([0.5,0.5, 0.5, 0.5]),
            cirq.PauliSum.from_pauli_strings([cirq.X(cirq.LineQubit(i)) for i in range(2)]),
            0,
            2,
        ),
        (
            np.array([0.5,0.5, 0.5, 0.5]),
            cirq.PauliSum.from_pauli_strings([cirq.Y(cirq.LineQubit(i)) for i in range(2)]),
            2,
            2,
        ),
    ],
)
def test_evaluate_simple(state, observables, variances, n):
    model = MockAbstractModel("LineQubit", n)
    model.set_simulator("cirq")
    variance_obj = Variance(model, observables, state)
    assert sum(abs(variance_obj.evaluate(_qubit_order={cirq.LineQubit(i): i for i in range(n)}) - variances)) < 1e-14

"""
def test_json():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    objective = Correlation(ising, "Z")
    print(objective)
    json = objective.to_json_dict()
    
    objective2 = Correlation.from_json_dict(json)
    
    assert (objective == objective2)

#############################################################
#                                                           #
#                    Assert tests                           #
#                                                           #
#############################################################
"""
def test_evaluate_assert():
    model = MockAbstractModel("LineQubit", 1)
    variance_obj = Variance(model,np.array([1,0]), cirq.Z(cirq.LineQubit(0)))
    with pytest.raises(AssertionError):
        variance_obj.evaluate() 
"""
def test_exception():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    with pytest.raises(AssertionError):
        assert Correlation(ising, "Foo")

def test_exception():
    ising = Ising("GridQubit", [1, 2], np.ones((0, 2)), np.ones((1, 1)), np.ones((1, 2)))
    ising.set_simulator("qsim")
    ising.set_circuit("qaoa", {"p": 5})
    obj = Correlation(ising, "Z")
    with pytest.raises(AssertionError):
        assert obj.evaluate(np.zeros(shape=(5, 5, 5)))
"""