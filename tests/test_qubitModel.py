"""
Test parent class AbstractModel;
    -test whether correct error messages occur
    -test whether initialisation set obj.qubits correctly 
        based on some examples
    -try to check test coverage?
"""
# external import
from typing import Dict
import cirq

import pytest
import numpy as np
import sympy
from timeit import default_timer
from qutlet.utilities.generic import flatten

# internal import
from qutlet.models import QubitModel


class DummyQubitModel(QubitModel):
    def __init__(self, qubits: int):
        super().__init__(qubits)

    def to_json_dict(self) -> Dict:
        pass

    def from_json_dict(self):
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
