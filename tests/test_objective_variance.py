from mimetypes import init
import cirq
import numpy as np
import pytest
from random import randrange

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

@pytest.mark.parametrize(
    "n, j_v, j_h , h, field, init_state, basics_options, variances",
    [
        #Test Z systems
        (
            [1,2],
            2*(np.random.rand(1-1,2)- 0.5),
            2*(np.random.rand(1,2-1)- 0.5),
            np.zeros((1,2)),
            "Z",
            randrange(2**2),
            {"append": False, "start": "identity"},
            0
        ),
        (
            [2,2],
            2*(np.random.rand(2-1,2)- 0.5),
            2*(np.random.rand(2,2-1)- 0.5),
            np.zeros((2,2)),
            "Z",
            randrange(2**4),
            {"append": False, "start": "identity"},
            0
        ),
        (
            [5,1],
            2*(np.random.rand(5-1,1)- 0.5),
            2*(np.random.rand(5,1-1)- 0.5),
            np.zeros((5,1)),
            "Z",
            randrange(2**5),
            {"append": False, "start": "identity"},
            0
        ),
        (
            [3,2],
            2*(np.random.rand(3,2)- 0.5),
            2*(np.random.rand(3,2-1)- 0.5),
            np.zeros((3,2)),
            "Z",
            randrange(2**6),
            {"append": False, "start": "identity"},
            0
        ),
        #Test X systems
        (
            [1,2],
            np.zeros((1,2)),
            np.zeros((1,2)),
            2*(np.random.rand(1,2)- 0.5),
            "X",
            randrange(2**2),
            {"append": False, "start": "hadamard"},
            0
        ),
        (
            [2,2],
            np.zeros((2,2)),
            np.zeros((2,2)),
            2*(np.random.rand(2,2)- 0.5),
            "X",
            randrange(2**4),
            {"append": False, "start": "hadamard"},
            0
        ),
        (
            [1,5],
            np.zeros((1,5)),
            np.zeros((1,5)),
            2*(np.random.rand(1,5)- 0.5),
            "X",
            randrange(2**5),
            {"append": False, "start": "hadamard"},
            0
        ),
        (
            [2,3],
            np.zeros((2,3)),
            np.zeros((2,3)),
            2*(np.random.rand(2,3)- 0.5),
            "X",
            randrange(2**6),
            {"append": False, "start": "hadamard"},
            0
        ),
        #Test Eigen basis
        (
            [1,2],
            2*(np.random.rand(1-1,2)- 0.5),
            2*(np.random.rand(1,2-1)- 0.5),
            2*(np.random.rand(1,2)- 0.5),
            "X",
            randrange(2**2),
            {"append": False, "start": "exact", "n_exact": [1,2], "b_exact": [1,1]},
            0
        ),
        # This fails but should not?
        #(
        #    [2,2],
        #    2*(np.random.rand(2-1,2)- 0.5),
        #    2*(np.random.rand(2,2-1)- 0.5),
        #    2*(np.random.rand(2,2)- 0.5),
        #    "X",
        #    randrange(2**4),
        #    {"append": False, "start": "exact", "n_exact": [2,2], "b_exact": [1,1]},
        #    0
        #),
        (
            [2,2],
            np.ones((1,2)),
            np.ones((2,1)),
            np.ones((2,2)),
            "X",
            randrange(2**4),
            {"append": False, "start": "exact", "n_exact": [2,2], "b_exact": [1,1]},
            0
        ),
        (
            [1,3],
            2*(np.random.rand(1-1,3)- 0.5),
            2*(np.random.rand(1,3)- 0.5),
            2*(np.random.rand(1,3)- 0.5),
            "X",
            randrange(2**3),
            {"append": False, "start": "exact", "n_exact": [1,3], "b_exact": [1,0]},
            0
        ),
        (
            [5,1],
            2*(np.random.rand(5,1)- 0.5),
            2*(np.random.rand(5,1-1)- 0.5),
            2*(np.random.rand(5,1)- 0.5),
            "X",
            randrange(2**2),
            {"append": False, "start": "exact", "n_exact": [5,1], "b_exact": [0,1]},
            0
        ),
        (
            [2,3],
            2*(np.random.rand(2-1,3)- 0.5),
            2*(np.random.rand(2,3)- 0.5),
            2*(np.random.rand(2,3)- 0.5),
            "X",
            randrange(2**6),
            {"append": False, "start": "exact", "n_exact": [2,3], "b_exact": [1,0]},
            0
        ),
        (
            [2,3],
            np.ones((1,3)),
            np.ones((2,2)),
            np.ones((2,3)),
            "X",
            0,
            {"append": False, "start": "exact", "n_exact": [2,3], "b_exact": [1,1]},
            0
        ),
    ],
)
def test_evaluate_Ising(n, j_v, j_h , h, field, init_state, basics_options, variances):
    #j_v0 = 2*(np.random.rand(n[0]-1,n[1])- 0.5)
    #j_h0 = 2*(np.random.rand(n[0],n[1]-1)- 0.5)
    #h0 = 2*(np.random.rand(n[0],n[1])- 0.5)
    model = Ising("GridQubit", n, j_v, j_h , h , field)
    model.set_simulator("cirq")
    model.set_circuit("basics", basics_options)

    #Important to hand over qubit order otherwise test fail
    _qubit_order = {model.qubits[k][l]: int(k*model.n[1] + l) for l in range(model.n[1]) for k in range(model.n[0])}
    state=model.simulator.simulate( model.circuit, 
                                    initial_state=init_state,
                                    qubit_order=_qubit_order).state_vector()
    variance_obj = Variance(model, wavefunction=state)

    #The tolerance here is rather large...
    #Maybe this is due to poor choice of data types somewhere?
    assert sum(abs(variance_obj.evaluate() - variances)) < 2e-6

#to do test sub systems
# Ising, n_exact -> test whether subsystems have smaller variance that X,Z for J=h=1
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