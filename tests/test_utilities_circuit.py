import numpy as np
import sympy


import cirq
import pytest

import qutlet.utilities.fermion
import qutlet.utilities.circuit


@pytest.mark.parametrize(
    "qubits, correct",
    [(cirq.GridQubit.rect(4, 4), (4, 4)), (cirq.LineQubit.range(10), (10, 1))],
)
def test_qubits_shape(qubits, correct):
    assert qutlet.utilities.circuit.qubits_shape(qubits=qubits) == correct


@pytest.mark.parametrize(
    "circuit, correct",
    [
        (cirq.Circuit(cirq.X(cirq.LineQubit(1))), 1),
        (cirq.Circuit((cirq.X(cirq.LineQubit(1)), cirq.Y(cirq.LineQubit(1)))), 2),
    ],
)
def test_depth(circuit, correct):
    assert qutlet.utilities.circuit.depth(circuit=circuit) == correct


@pytest.mark.parametrize(
    "pstr,anti,correct",
    [
        (
            cirq.PauliString(
                -1j,
                cirq.X(cirq.LineQubit(0)),
                cirq.Y(cirq.LineQubit(1)),
                cirq.Z(cirq.LineQubit(2)),
            ),
            True,
            True,
        ),
        (
            cirq.PauliString(
                221j,
                cirq.Y(cirq.LineQubit(0)),
                cirq.Z(cirq.LineQubit(1)),
                cirq.X(cirq.LineQubit(2)),
            ),
            False,
            False,
        ),
    ],
)
def test_pauli_str_is_hermitian(pstr, anti, correct):
    assert qutlet.utilities.circuit.pauli_str_is_hermitian(pstr, anti) == correct


@pytest.mark.parametrize(
    "psum,anti,correct",
    [
        (
            cirq.X(cirq.LineQubit(0))
            + cirq.Y(cirq.LineQubit(1))
            + cirq.Z(cirq.LineQubit(2)),
            False,
            True,
        ),
        (
            1j * cirq.Y(cirq.LineQubit(0))
            + cirq.Z(cirq.LineQubit(1))
            + cirq.X(cirq.LineQubit(2)),
            True,
            False,
        ),
    ],
)
def test_pauli_sum_is_hermitian(psum, anti, correct):
    assert qutlet.utilities.circuit.pauli_sum_is_hermitian(psum, anti) == correct


@pytest.mark.parametrize(
    "pstr,anti,correct",
    [
        (
            cirq.PauliString(
                -1j,
                cirq.X(cirq.LineQubit(0)),
                cirq.Y(cirq.LineQubit(1)),
                cirq.Z(cirq.LineQubit(2)),
            ),
            False,
            cirq.PauliString(
                1,
                cirq.X(cirq.LineQubit(0)),
                cirq.Y(cirq.LineQubit(1)),
                cirq.Z(cirq.LineQubit(2)),
            ),
        ),
        (
            cirq.PauliString(
                1 + 221j,
                cirq.Y(cirq.LineQubit(0)),
            ),
            True,
            cirq.PauliString(
                221j,
                cirq.Y(cirq.LineQubit(0)),
            ),
        ),
        (
            cirq.PauliString(
                1 + 22j,
                cirq.Y(cirq.LineQubit(0)),
            ),
            False,
            cirq.PauliString(
                1,
                cirq.Y(cirq.LineQubit(0)),
            ),
        ),
        (
            cirq.PauliString(
                1,
                cirq.Y(cirq.LineQubit(0)),
            ),
            False,
            cirq.PauliString(
                1,
                cirq.Y(cirq.LineQubit(0)),
            ),
        ),
    ],
)
def test_make_pauli_str_hermitian(pstr, anti, correct):
    assert qutlet.utilities.circuit.make_pauli_str_hermitian(pstr, anti) == correct


@pytest.mark.parametrize(
    "psum,anti,correct",
    [
        (
            cirq.X(cirq.LineQubit(0))
            + cirq.Y(cirq.LineQubit(1))
            + cirq.Z(cirq.LineQubit(2)),
            True,
            1j
            * (
                cirq.X(cirq.LineQubit(0))
                + cirq.Y(cirq.LineQubit(1))
                + cirq.Z(cirq.LineQubit(2))
            ),
        ),
        (
            cirq.Y(cirq.LineQubit(0)) + cirq.Z(cirq.LineQubit(1)),
            False,
            cirq.Y(cirq.LineQubit(0)) + cirq.Z(cirq.LineQubit(1)),
        ),
        (
            cirq.Y(cirq.LineQubit(0)) + cirq.Z(cirq.LineQubit(1)),
            True,
            1j * cirq.Y(cirq.LineQubit(0)) + 1j * cirq.Z(cirq.LineQubit(1)),
        ),
    ],
)
def test_make_pauli_sum_hermitian(psum, anti, correct):
    h_psum = qutlet.utilities.circuit.make_pauli_sum_hermitian(psum, anti)
    assert h_psum == correct


@pytest.mark.parametrize(
    "pstr,correct",
    [
        (cirq.PauliString(*(cirq.I(cirq.LineQubit(x)) for x in range(10))), True),
        (
            cirq.PauliString(
                *(cirq.X(cirq.LineQubit(x)) for x in range(3)),
                cirq.X(cirq.LineQubit(29))
            ),
            False,
        ),
    ],
)
def test_pauli_str_is_identity(pstr, correct):
    assert qutlet.utilities.circuit.pauli_str_is_identity(pstr) == correct


def test_pauli_str_is_identity_err():
    with pytest.raises(ValueError):
        qutlet.utilities.circuit.pauli_str_is_identity(0)


@pytest.mark.parametrize(
    "psum,correct",
    [
        (
            cirq.X(cirq.LineQubit(0)) * cirq.X(cirq.LineQubit(1))
            + cirq.Y(cirq.LineQubit(1))
            + cirq.Z(cirq.LineQubit(2)),
            False,
        ),
        (
            cirq.X(cirq.LineQubit(0))
            + cirq.X(cirq.LineQubit(1))
            + cirq.X(cirq.LineQubit(2)),
            True,
        ),
    ],
)
def test_all_pauli_str_commute(psum, correct):
    assert qutlet.utilities.circuit.all_pauli_str_commute(psum) == correct


@pytest.mark.parametrize(
    "pauli,correct",
    [
        (
            cirq.X(cirq.LineQubit(0))
            * cirq.X(cirq.LineQubit(1))
            * cirq.Y(cirq.LineQubit(3))
            * cirq.Z(cirq.LineQubit(4)),
            cirq.Circuit(
                [
                    cirq.H(cirq.LineQubit(0)),
                    cirq.H(cirq.LineQubit(1)),
                    cirq.S(cirq.LineQubit(3)),
                    cirq.H(cirq.LineQubit(3)),
                ]
            ),
        ),
    ],
)
def test_pauli_basis_change(pauli, correct):
    assert qutlet.utilities.circuit.pauli_basis_change(pauli) == correct
