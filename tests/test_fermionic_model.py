import pytest
import openfermion as of
from models.fermionic_model import FermionicModel
from qutlet.utilities import jw_computational_wf, index_bits


@pytest.mark.parametrize(
    "n_qubits,bitstring,correct",
    [
        (8, "01011011", [2, 3]),
        (7, "0101011", [1, 3]),
        (4, "0000", [0, 0]),
        (3, "001", [1, 0]),
        (4, "0001", [0, 1]),
    ],
)
def test_spin_and_number_operator(n_qubits, bitstring, correct):
    n_up_op, n_down_op, n_total_op = FermionicModel.spin_and_number_operator(
        n_qubits=n_qubits
    )
    state = jw_computational_wf(index_bits(bitstring), n_qubits, reverse=True)
    n_up = int(
        of.expectation(of.get_sparse_operator(n_up_op, n_qubits=n_qubits), state)
    )
    n_down = int(
        of.expectation(of.get_sparse_operator(n_down_op, n_qubits=n_qubits), state)
    )
    n_total = int(
        of.expectation(of.get_sparse_operator(n_total_op, n_qubits=n_qubits), state)
    )
    assert n_up == correct[0]
    assert n_down == correct[1]
    assert n_total == sum(correct)
