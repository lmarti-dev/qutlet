"""
Test parent class AbstractModel;
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
from fauvqe import AbstractModel


class MockAbstractModel(AbstractModel):
    def to_json_dict(self):
        return {}

    def from_json_dict(self, dct):
        return MockInitialiser()

    def energy(self):
        return np.array([])


# test_AbstractModel_set_qubits
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
def test_AbstractModel(qubittype, n, exp_quibits):
    AbstractModel_obj = MockAbstractModel(qubittype, n)
    assert AbstractModel_obj.qubits == exp_quibits
    assert AbstractModel_obj.qubittype == qubittype
    if isinstance(n, np.ndarray):
        assert (AbstractModel_obj.n == n).all()
    else:
        assert AbstractModel_obj.n == n


# test whether circuit and simulator was created
def test_AbstractModel_exist():
    AbstractModel_obj = MockAbstractModel("LineQubit", 1)
    assert hasattr(AbstractModel_obj, "simulator")
    assert hasattr(AbstractModel_obj, "qubits")


# test whether circuit and simulator was created
def test_set_simulator():
    AbstractModel_obj = MockAbstractModel("LineQubit", 1)
    # Check if default parameter is given
    assert type(AbstractModel_obj.simulator) == qsimcirq.qsim_simulator.QSimSimulator
    assert AbstractModel_obj.simulator_options == {"t": 8, "f": 4}
    # Check whether adding parameters works for qsim
    AbstractModel_obj.set_simulator(simulator_options={"f": 2})
    assert AbstractModel_obj.simulator_options == {"t": 8, "f": 2}

    # Check whether cirq simulator can be set
    # and whether simulator_options are correct default
    AbstractModel_obj.set_simulator(simulator_name="cirq")
    assert type(AbstractModel_obj.simulator) == cirq.sim.sparse_simulator.Simulator
    assert AbstractModel_obj.simulator_options == {}

    # Test whether an Assertion error is raised otherwise
    with pytest.raises(AssertionError):
        AbstractModel_obj.set_simulator(simulator_name="simulator")
