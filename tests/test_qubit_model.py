"""
Test parent class QubitModel;
    -test whether correct error messages occur
    -test whether initialisation set obj.qubits correctly 
        based on some examples
    -try to check test coverage?
"""

# external import
from typing import Dict
import cirq

import pytest  # noqa

# internal import
from qutlet.models import QubitModel


class DummyQubitModel(QubitModel):
    def __init__(self, qubits: int):
        super().__init__(qubits)

    def __to_json__(self) -> Dict:
        pass


def test_init():
    qm = DummyQubitModel(qubits=(2, 5))
    assert qm[1, 4] == qm[9]
    assert qm[0, 4] == qm[4]


def test_props():
    qm = DummyQubitModel(qubits=3)
    assert qm.qmap == {val: ind for ind, val in enumerate(qm._qubits)}
    assert qm.qid_shape == (2, 2, 2)
    assert qm.qubits == [cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2)]
