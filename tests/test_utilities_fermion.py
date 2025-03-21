import numpy as np
from scipy.sparse import csc_matrix

from cirq import allclose_up_to_global_phase
import openfermion as of
import pytest

import qutlet.utilities.testing
import qutlet.utilities.fermion
from qutlet.models import FermiHubbardModel


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
    ee_fop = qutlet.utilities.fermion.even_excitation(
        coeff=coeff, indices=indices, anti_hermitian=anti_hermitian
    )
    assert ee_fop == correct_fop


def test_even_excitation_error():
    with pytest.raises(ValueError):
        qutlet.utilities.fermion.even_excitation(
            coeff=1, indices=[1, 2, 3, 4, 5], anti_hermitian=True
        )


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
    single_fop = qutlet.utilities.fermion.single_excitation(
        coeff, i, j, anti_hermitian=anti_hermitian
    )
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
    double_fop = qutlet.utilities.fermion.double_excitation(
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
    assert qutlet.utilities.fermion.bravyi_kitaev_fast_wrapper(fop) == correct_fop


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
    assert np.isclose(qutlet.utilities.fermion.mean_coeff_n_terms(fop, n), correct)


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
    assert qutlet.utilities.testing.do_lists_have_same_elements(
        qutlet.utilities.fermion.jw_spin_correct_indices(n_electrons, n_indices),
        correct,
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
        qutlet.utilities.fermion.jw_spin_restrict_operator(
            sparse_operator, particle_number, n_qubits
        )
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
            csc_matrix(np.eye(16, k=2) + np.eye(16, k=-2)),
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
def test_jw_eigenspectrum_at_particle_number(
    sparse_operator,
    particle_number,
    expanded,
    spin,
    sparse,
    k,
    correct_eigvals,
    correct_eigvecs,
):
    eigvals, eigvecs = qutlet.utilities.fermion.jw_eigenspectrum_at_particle_number(
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
    (
        eigval,
        eigvec,
    ) = qutlet.utilities.fermion.jw_get_true_ground_state_at_particle_number(
        sparse_operator, particle_number, spin, sparse
    )
    assert np.isclose(eigval, correct_eigval).all()
    assert np.isclose(eigvec, correct_eigvec).all()


######################################################################
#                           Errors                                   #
######################################################################


def test_jw_spin_correct_indices_error():
    with pytest.raises(TypeError):
        qutlet.utilities.fermion.jw_spin_correct_indices("hello", [1, 2])


@pytest.mark.parametrize(
    "indices, Nqubits, correct",
    [
        ([1, 2], 3, np.array([0, 0, 0, 0, 0, 0, 1, 0])),
        ([0, 1], 2, np.array([0, 0, 0, 1])),
    ],
)
def test_jw_computational_wf(indices, Nqubits, correct):
    assert (
        qutlet.utilities.fermion.jw_computational_wf(indices, Nqubits) == correct
    ).all()


# can't properly test since the order of the bogos is not the same as
# the full eigvecs and I don't want to do a double for loop with elimination
# def test_jw_free_couplers():
#     model = FermiHubbardModel(
#         lattice_dimensions=(2, 2), tunneling=1, coulomb=2, n_electrons=[2, 2]
#     )

#     free_couplers = qutlet.utilities.fermion.jw_get_free_couplers(model)

#     _, eigvecs = model.non_interacting_model.spectrum
#     matrix_couplers = [
#         np.multiply.outer(eigvecs[:, 0].conjugate(), eigvecs[:, k])
#         for k in range(1,len(eigvecs[0, :]))
#     ]

#     for fc, mc in zip(free_couplers, matrix_couplers):
#         fcm = fc.matrix(qubits=model.qubits)
#         assert allclose_up_to_global_phase(fcm, mc)


def test_bogo_creators():
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2), tunneling=1, coulomb=2, n_electrons=[2, 2]
    )

    matrix = qutlet.utilities.fermion.get_bogoliubov_matrix(
        of.get_fermion_operator(model.quadratic_terms), model.n_qubits
    )

    sector_matrix_up = matrix[::2, ::2]
    _, eigvecs_up = np.linalg.eigh(sector_matrix_up)
    sector_matrix_down = matrix[1::2, 1::2]
    _, eigvecs_down = np.linalg.eigh(sector_matrix_down)

    bogos_up = qutlet.utilities.fermion.get_bogo_fop(eigvecs_up.T, 0)
    bogos_down = qutlet.utilities.fermion.get_bogo_fop(eigvecs_down.T, 1)

    bogo_creators = qutlet.utilities.fermion.build_bogo_creators(
        bogos_up, bogos_down, model.n_electrons
    )

    assert len(bogo_creators) == qutlet.utilities.fermion.get_fermionic_subspace_size(
        model.n_electrons, model.n_qubits
    )

    free_energies, _ = model.non_interacting_model.spectrum

    vacuum = np.zeros((2**model.n_qubits,))
    vacuum[0] = 1

    bogo_energies = np.zeros_like(free_energies)
    for ind, bogo_creator in enumerate(bogo_creators):
        slater_state = (
            of.get_sparse_operator(bogo_creator, n_qubits=model.n_qubits).toarray()
            @ vacuum
        )
        bogo_energies[ind] = model.non_interacting_model.statevector_expectation(
            slater_state
        )

    assert np.isclose(np.sort(bogo_energies), free_energies).all()
