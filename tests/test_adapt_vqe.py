import pytest
import cirq
from fauvqe.models.circuits import adapt_vqe as advqe
from fauvqe.utilities import dicke_state, jw_computational_wf


def test_fidelity_gradient():
    qubits = cirq.LineQubit.range(8)
    trial_state = dicke_state(n=8, k=4)
    target_state = jw_computational_wf(indices=[0, 2, 3], Nqubits=8)
    paulisum = cirq.X(qubits[2]) * cirq.X(qubits[3])
    op = paulisum.matrix(qubits=qubits)
    advqe.compute_fidelity_gradient(
        op, trial_state=trial_state, target_state=target_state
    )
