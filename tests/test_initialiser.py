"""
Test parent class Initialiser;
    -test whether correct error messages occur
    -test whether initialisation set obj.qubits correctly 
        based on some examples
    -try to check test coverage?
"""
# external import
import cirq
import qsimcirq
import pytest
import numpy as np

# internal import
from fauvqe import Initialiser


class MockInitialiser(Initialiser):
    def energy(self):
        return np.array([])


# test_Initialiser_set_qubits
@pytest.mark.parametrize(
    "qubittype, n, exp_quibits",
    [
        ("NamedQubit", "a", [cirq.NamedQubit("a")]),
        ("NamedQubit", ["a", "b"], [cirq.NamedQubit("a"), cirq.NamedQubit("b")]),
        ("LineQubit", 1, [cirq.LineQubit(0)]),
        # This should work but doesn't:
        # ('LineQubit', np.array(1), [cirq.LineQubit(0)]),
        ("LineQubit", 2, [cirq.LineQubit(0), cirq.LineQubit(1)]),
        ("GridQubit", np.array([1, 1]), [[cirq.GridQubit(0, 0)]]),
        ("GridQubit", [1, 1], [[cirq.GridQubit(0, 0)]]),
        ("GridQubit", [2, 1], [[cirq.GridQubit(0, 0)], [cirq.GridQubit(1, 0)]]),
        ("GridQubit", [1, 2], [[cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]]),
    ],
)
def test_Initialiser(qubittype, n, exp_quibits):
    initialiser_obj = MockInitialiser(qubittype, n)
    assert initialiser_obj.qubits == exp_quibits
    assert initialiser_obj.qubittype == qubittype
    if isinstance(n, np.ndarray):
        assert (initialiser_obj.n == n).all()
    else:
        assert initialiser_obj.n == n


# test whether circuit and simulator was created
def test_Initialiser_exist():
    initialiser_obj = MockInitialiser("LineQubit", 1)
    assert hasattr(initialiser_obj, "simulator")
    assert hasattr(initialiser_obj, "qubits")


# test whether circuit and simulator was created
def test_set_simulator():
    initialiser_obj = MockInitialiser("LineQubit", 1)
    # Check if default parameter is given
    assert type(initialiser_obj.simulator) == qsimcirq.qsim_simulator.QSimSimulator
    assert initialiser_obj.simulator_options == {"t": 8, "f": 4}
    # Check whether adding parameters works for qsim
    initialiser_obj.set_simulator(simulator_options={"f": 2})
    assert initialiser_obj.simulator_options == {"t": 8, "f": 2}

    # Check whether cirq simulator can be set
    # and whether simulator_options are correct default
    initialiser_obj.set_simulator(simulator_name="cirq")
    assert type(initialiser_obj.simulator) == cirq.sim.sparse_simulator.Simulator
    assert initialiser_obj.simulator_options == {}

    # Test whether an Assertion error is raised otherwise
    with pytest.raises(AssertionError):
        initialiser_obj.set_simulator(simulator_name="simulator")
