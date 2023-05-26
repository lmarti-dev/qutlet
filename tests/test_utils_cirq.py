import pytest
import openfermion as of
import fauvqe.utils_cirq as cqutils
import fauvqe.utils as utils
from fauvqe.models.ising import Ising
import numpy as np
import cirq
from scipy.sparse import csc_matrix
import sympy


@pytest.mark.parametrize(
    "coeff, indices, anti_hermitian, correct",
    [
        (0.5, [1, 2], False, [(1, 1), ("1^ 2", "2^ 1")]),
        (0.5, [7, 7], True, [(1j,), ("7^ 7",)]),
        (0.666, [1, 2, 3, 4], False, [(1, 1), (("1^ 2^ 3 4", "4^ 3^ 2 1"))]),
        (
            0.65,
            [1, 2, 3, 4, 5, 6],
            False,
            [(1, 1), ("1^ 2^ 3^ 4 5 6", "6^ 5^ 4^ 3 2 1")],
        ),
        (
            0.65,
            [1, 2, 3, 4, 5, 6, 7, 8],
            True,
            [(1, -1), ("1^ 2^ 3^ 4^ 5 6 7 8", "8^ 7^ 6^ 5^ 4 3 2 1")],
        ),
        (
            0.65,
            [2, 10, 33, 21, 12, 3],
            True,
            [(1, -1), ("2^ 10^ 33^ 21 12 3", "3^ 12^ 21^ 33 10 2")],
        ),
        (
            0.65,
            [3, 3],
            False,
            [(1,), ("3^ 3",)],
        ),
    ],
)
def test_even_excitation(coeff, indices, anti_hermitian, correct):
    correct_fop = sum(
        [
            coeff * parity * of.FermionOperator(s)
            for parity, s in zip(correct[0], correct[1])
        ]
    )
    ee_fop = cqutils.even_excitation(
        coeff=coeff, indices=indices, anti_hermitian=anti_hermitian
    )
    assert ee_fop == correct_fop


def test_even_excitation_error():
    with pytest.raises(ValueError):
        cqutils.even_excitation(coeff=1, indices=[1, 2, 3, 4, 5], anti_hermitian=True)


@pytest.mark.parametrize(
    "coeff, i,j, anti_hermitian, correct",
    [
        (0.5, 1, 2, False, [(1, 1), ("1^ 2", "2^ 1")]),
        (0.5, 7, 7, True, [(1j,), ("7^ 7",)]),
    ],
)
def test_single_excitation(coeff, i, j, anti_hermitian, correct):
    correct_fop = sum(
        [
            coeff * parity * of.FermionOperator(s)
            for parity, s in zip(correct[0], correct[1])
        ]
    )
    single_fop = cqutils.single_excitation(coeff, i, j, anti_hermitian=anti_hermitian)
    assert single_fop == correct_fop


@pytest.mark.parametrize(
    "coeff, i,j,k,l, anti_hermitian, correct",
    [
        (0.666, 7, 200, 3, 4, False, [(1, 1), (("7^ 200^ 3 4", "4^ 3^ 200 7"))]),
        (0.6j, 12, 200, 3, 4, True, [(1, 1), (("12^ 200^ 3 4", "4^ 3^ 200 12"))]),
    ],
)
def test_double_excitation(coeff, i, j, k, l, anti_hermitian, correct):
    correct_fop = sum(
        [
            coeff * parity * of.FermionOperator(s)
            for parity, s in zip(correct[0], correct[1])
        ]
    )
    double_fop = cqutils.double_excitation(
        coeff, i, j, k, l, anti_hermitian=anti_hermitian
    )
    assert double_fop == correct_fop


@pytest.mark.parametrize(
    "fop, correct",
    [
        (
            of.FermionOperator("1^ 2"),
            (
                0.0,
                [
                    [
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                    ],
                    [0, 1, 0],
                ],
                np.zeros((3, 3, 3, 3)),
            ),
        ),
    ],
)
def test_bravyi_kitaev_fast_wrapper(fop, correct):
    correct_fop = of.bravyi_kitaev_fast(
        of.InteractionOperator(
            constant=correct[0],
            one_body_tensor=np.array(correct[1]),
            two_body_tensor=np.array(correct[2]),
        )
    )
    assert cqutils.bravyi_kitaev_fast_wrapper(fop) == correct_fop


@pytest.mark.parametrize(
    "fop, n, correct",
    [
        (
            of.FermionOperator("0^ 3", 2)
            + of.FermionOperator("2^ 3", 1)
            + of.FermionOperator("2^ 3 4^ 6", 1000),
            2,
            1.5,
        ),
        (
            of.FermionOperator("0^ 3", 2)
            + of.FermionOperator("2^ 3", 1)
            + of.FermionOperator("2^ 3 4^ 6", 1000)
            + of.FermionOperator("9^ 3 4^ 6", 2000),
            4,
            1500,
        ),
    ],
)
def test_mean_coeff_n_terms(fop, n, correct):
    assert np.isclose(cqutils.mean_coeff_n_terms(fop, n), correct)


@pytest.mark.parametrize(
    "qubits, correct",
    [(cirq.GridQubit.rect(4, 4), (4, 4)), (cirq.LineQubit.range(10), (10, 1))],
)
def test_qubits_shape(qubits, correct):
    assert cqutils.qubits_shape(qubits=qubits) == correct


@pytest.mark.parametrize(
    "circuit, correct",
    [
        (cirq.Circuit(cirq.X(cirq.LineQubit(1))), 1),
        (cirq.Circuit((cirq.X(cirq.LineQubit(1)), cirq.Y(cirq.LineQubit(1)))), 2),
    ],
)
def test_depth(circuit, correct):
    assert cqutils.depth(circuit=circuit) == correct


@pytest.mark.parametrize(
    "n_electrons,n_indices, correct",
    [
        (2, 4, [3, 6, 9, 12]),
        ([1, 2], 6, [11, 35, 41, 14, 38, 44, 26, 50, 56]),
        (
            1,
            2,
            [
                1,
            ],
        ),
    ],
)
def test_jw_spin_correct_indices(n_electrons, n_indices, correct):
    assert utils.lists_have_same_elements(
        cqutils.jw_spin_correct_indices(n_electrons, n_indices), correct
    )


@pytest.mark.parametrize(
    "sparse_operator, particle_number, n_qubits,correct",
    [
        (csc_matrix(np.reshape(np.arange(16), (4, 4))), 1, 2, csc_matrix([[5]])),
        (
            csc_matrix(np.reshape(np.arange(16**2), (16, 16))),
            [1, 2],
            None,
            csc_matrix([[11 * 16 + 11, 11 * 16 + 14], [14 * 16 + 11, 14 * 16 + 14]]),
        ),
    ],
)
def test_jw_spin_restrict_operator(sparse_operator, particle_number, n_qubits, correct):
    assert (
        cqutils.jw_spin_restrict_operator(sparse_operator, particle_number, n_qubits)
        != correct
    ).nnz == 0


@pytest.mark.parametrize(
    "sparse_operator, particle_number, expanded, spin, sparse, k,correct_eigvals,correct_eigvecs",
    [
        (
            csc_matrix(np.eye(4, k=1) + np.eye(4, k=-1)),
            [0, 1],
            True,
            False,
            True,
            1,
            [-1],
            [[0], [-0.70710678], [0.70710678], [0]],
        ),
        (
            csc_matrix(np.eye(16, k=3) + np.eye(16, k=-3)),
            [1, 0],
            False,
            True,
            False,
            None,
            [-1.0, 1.0],
            [[-0.70710678, 0.70710678], [0.70710678, 0.70710678]],
        ),
    ],
)
def test_eigenspectrum_at_particle_number(
    sparse_operator,
    particle_number,
    expanded,
    spin,
    sparse,
    k,
    correct_eigvals,
    correct_eigvecs,
):
    eigvals, eigvecs = cqutils.eigenspectrum_at_particle_number(
        sparse_operator,
        particle_number,
        expanded,
        spin,
        sparse,
        k,
    )
    assert np.isclose(eigvals, correct_eigvals).all()
    assert np.isclose(eigvecs, correct_eigvecs).all()


@pytest.mark.parametrize(
    "sparse_operator, particle_number, spin, sparse,correct_eigval,correct_eigvec",
    [
        (
            csc_matrix(np.eye(4, k=1) + np.eye(4, k=-1)),
            [0, 1],
            False,
            False,
            -1,
            [0, -0.70710678, 0.70710678, 0],
        ),
    ],
)
def test_jw_get_true_ground_state_at_particle_number(
    sparse_operator, particle_number, spin, sparse, correct_eigval, correct_eigvec
):
    eigval, eigvec = cqutils.jw_get_true_ground_state_at_particle_number(
        sparse_operator, particle_number, spin, sparse
    )
    assert np.isclose(eigval, correct_eigval).all()
    assert np.isclose(eigvec, correct_eigvec).all()


def test_get_param_resolver():
    model = Ising("GridQubit", (1, 1))
    sym = sympy.Symbol("x")
    model.circuit_param = []
    model.circuit_param.append(sym)
    model.circuit_param_values = []
    model.circuit_param_values.append(1)

    assert cqutils.get_param_resolver(
        model=model, param_values=model.circuit_param_values
    ) == cirq.ParamResolver({str(sym): 1})


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
    assert cqutils.pauli_str_is_hermitian(pstr, anti) == correct


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
    assert cqutils.pauli_sum_is_hermitian(psum, anti) == correct


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
    assert cqutils.make_pauli_str_hermitian(pstr, anti) == correct


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
    h_psum = cqutils.make_pauli_sum_hermitian(psum, anti)
    assert h_psum == correct


def test_qmap():
    model = Ising("GridQubit", (10, 1))
    qs = cirq.GridQubit.rect(10, 1)
    assert cqutils.qmap(model) == {qs[x]: x for x in range(10)}


def test_populate_empty_qubits():
    model = Ising("GridQubit", (10, 1))
    circ = cirq.Circuit([cirq.I(mq) for mq in model.flattened_qubits])
    assert circ == cqutils.populate_empty_qubits(model)


def test_match_param_values_to_symbols():
    model = Ising("GridQubit", (10, 1))
    model.circuit_param_values = None
    symbols = (sympy.Symbol("a"), sympy.Symbol("b"))
    cqutils.match_param_values_to_symbols(model=model, symbols=symbols)
    assert (model.circuit_param_values == np.zeros(len(symbols))).all()


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
    assert cqutils.pauli_str_is_identity(pstr) == correct


def test_pauli_str_is_identity_err():
    with pytest.raises(ValueError):
        cqutils.pauli_str_is_identity(0)


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
    assert cqutils.all_pauli_str_commute(psum) == correct


######################################################################
#                           Errors                                   #
######################################################################


def test_jw_spin_correct_indices_error():
    with pytest.raises(TypeError):
        cqutils.jw_spin_correct_indices("hello", [1, 2])
