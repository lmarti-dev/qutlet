import itertools
from typing import List, Tuple, Union, TYPE_CHECKING
import numpy as np
import openfermion as of
from cirq import PauliSum

import scipy
from qutlet.utilities.generic import sum_even, sum_odd, index_bits, binleftpad

if TYPE_CHECKING:
    from qutlet.models import FermionicModel


def get_fermionic_states_number(n_electrons: list, n_qubits: int):
    return len(
        list(itertools.combinations(range(n_qubits // 2), n_electrons[0]))
    ) * len(list(itertools.combinations(range(n_qubits // 2), n_electrons[1])))


def even_excitation(
    coeff: float, indices: List[int], anti_hermitian: bool
) -> of.FermionOperator:
    """Returns an even fock operator of the type a*i a*j a*k al am an
    Args:
        coeff (float): the coefficient
        indices (List[int]): the indices of the annihilation and creation operators
        anti_hermitian (bool): whether to make the operator antihermiatian
    Raises:
        ValueError: if the index array length is odd
    Returns:
        of.FermionOperator: the fock operator
    """
    if len(indices) % 2 != 0:
        raise ValueError(
            "expected an even length for the indices list but got: {}".format(
                len(indices)
            )
        )

    half_len = int(len(indices) / 2)
    # split the indices between annihilation and creation parts
    iind = indices[:half_len]
    jind = indices[half_len:]
    # ai*aj*akal
    ac1 = ["{}^".format(n) for n in iind]
    aa1 = ["{}".format(n) for n in jind]
    # get full op
    op = of.FermionOperator(" ".join(ac1 + aa1), coefficient=coeff)
    # manage anti-hermitianicity
    # get h.c
    opdagg = of.hermitian_conjugated(op)
    if op == opdagg:
        # if we're already have an hermitian operator eg 7^ 7
        if anti_hermitian:
            # avoid doing op - op = 0 in this case
            return 1j * op
        return op
    else:
        if anti_hermitian:
            return op - opdagg
        return op + opdagg


def fermion_op_sites_number(fop: of.FermionOperator) -> int:
    return max([z[0] for y in fop.terms.keys() for z in y])


def double_excitation(
    coeff: float, i: int, j: int, k: int, l: int, anti_hermitian: bool  # noqa
) -> of.FermionOperator:
    """Return a fock operator of the type a*i a*j ak al and makes it hermitian or anti-hermitian
    Args:
        coeff (float): the coefficient
        i (int): the first index
        j (int): the second index
        k (int): the third index
        l (int): the fourth index
        anti_hermitian (bool): whether to make the operator anti-hermitian
    Returns:
        of.FermionOperator: the fock operator
    """
    return even_excitation(
        coeff=coeff, indices=[i, j, k, l], anti_hermitian=anti_hermitian
    )


def single_excitation(
    coeff: float, i: int, j: int, anti_hermitian: bool
) -> of.FermionOperator:
    """Returns a fock operator of the type a*i aj +- aj a*i
    Args:
        coeff (float): the coefficient
        i (int): the first index
        j (int): the second index
        anti_hermitian (bool): whether to make it anti-hermitian
    Returns:
        of.FermionOperator: the fock operator
    """
    return even_excitation(coeff=coeff, indices=[i, j], anti_hermitian=anti_hermitian)


def jw_spin_correct_indices(
    n_electrons: Union[list, int],
    n_qubits: int,
    right_to_left: bool = False,
) -> list:
    """Get the indices corresponding to the spin sectors given by n_electrons
    Args:
        n_electrons (Union[list, int]): the n_electrons in the system, or the spin sectors
        n_qubits (int): the number of qubits in the system
    Raises:
        TypeError: if the n_electrons is neither a list nor an int
    Returns:
        list: the indices associated with the correct number of spins/electrons
    """
    # since we usually fill even then odd indices, and in the default jw,
    # the up spins are even and odd are down, we check if we have the correct
    # up and down spin count before passing the indices back
    # when n_electrons is odd, we assume there is one more even indices, since those are filled even-first.
    # I guess we could use abs() but then we would mix two unrelated parity states
    # in the case n_spin_up is defined, then we count the number of spin_up (even indices) in the comb

    if isinstance(n_electrons, int):
        # we get the correct combinations by checking there is the same amount of even and odd indices in the comb
        n_spin_up = int(np.ceil(n_electrons / 2))
        n_spin_down = int(np.floor(n_electrons / 2))
    elif isinstance(n_electrons, list):
        n_spin_up = n_electrons[0]
        n_spin_down = n_electrons[1]
    else:
        raise TypeError(
            "Expected n_electrons to be either a list or an int, got {}".format(
                type(n_electrons)
            )
        )
    combinations = itertools.combinations(
        range(n_qubits), sum([n_spin_up, n_spin_down])
    )
    correct_combinations = [
        list(c)
        for c in combinations
        if sum_even(c) == n_spin_up and sum_odd(c) == n_spin_down
    ]

    if right_to_left:
        jw_indices = [
            sum([2 ** (n_qubits - 1 - iii) for iii in combination])
            for combination in correct_combinations
        ]
    else:
        jw_indices = [
            sum([2**iii for iii in combination]) for combination in correct_combinations
        ]
    return jw_indices


def jw_spin_restrict_operator(
    sparse_operator: scipy.sparse.csc_matrix,
    particle_number: Union[list, int],
    n_qubits: int,
    right_to_left: bool = False,
) -> scipy.sparse.csc_matrix:
    """Restrict a sparse operator to the subspace which contains only the allowed particle number
    Args:
        sparse_operator (scipy.sparse.csc_matrix): the hamiltonian in sparse representation
        particle_number (Union[list, int]): the particle number, or the spin sectors
        n_qubits (int): the number of qubits in the system (if not provided, will be computed from the matrix)
    Returns:
        scipy.sparse.csc_matrix: the restricted operator
    """
    if n_qubits is None:
        n_qubits = int(np.log2(sparse_operator.shape[0]))

    select_indices = jw_spin_correct_indices(
        n_electrons=particle_number, n_qubits=n_qubits, right_to_left=right_to_left
    )
    return sparse_operator[np.ix_(select_indices, select_indices)]


def jw_eigenspectrum_at_particle_number(
    sparse_operator: scipy.sparse.csc_matrix,
    particle_number: Union[int, list],
    expanded=False,
    spin: bool = True,
    sparse: bool = False,
    k: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the eigenspectrum (energies and wavefunctions) of some jw encoded fock hamiltonian
    Args:
        sparse_operator (scipy.sparse.csc_matrix): the sparse representation of the hamiltonian
        particle_number (Union[int | list]): the particle number (if a list, the up and down spin sectors)
        expanded (bool, optional): whether to expand back the vectors to the full size of the system, rather than the size of the sector corresponding to the particle number. Defaults to False.
        spin (bool, optional): whether to consider spin when looking at indices. Defaults to True.
        sparse (bool, optional): whether to use a sparse solver. Defaults to False.
        k (int, optional): if using a sparse solver, the number of eigenstates to find (must be smaller than the total number of eigenstates). Defaults to None.
    Returns:
        Tuple[np.ndarray,np.ndarray]: eigenvalues, eigenvectors
    """

    n_qubits = int(np.log2(sparse_operator.shape[0]))
    # Get the operator restricted to the subspace of the desired particle number

    kwargs = {}
    if spin:
        restricted_operator_func = jw_spin_restrict_operator
        indices_func = jw_spin_correct_indices
        kwargs.update({"right_to_left": True})
    else:
        # if you specified spin sectors but want to ignore spin, merge sectors
        if isinstance(particle_number, list):
            particle_number = sum(particle_number)
        restricted_operator_func = of.jw_number_restrict_operator
        indices_func = of.jw_number_indices
    restricted_operator = restricted_operator_func(
        sparse_operator, particle_number, n_qubits, **kwargs
    )
    # Compute eigenvalues and eigenvectors
    if restricted_operator.size == 1:
        print(
            "Restricted operator has shape {}, not solving sparsely".format(
                restricted_operator.shape
            )
        )
        sparse = False
    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(
            restricted_operator, k=k, which="SA"
        )
    else:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    if expanded:
        n_eigvecs = eigvecs.shape[-1]
        expanded_eigvecs = np.zeros((2**n_qubits, n_eigvecs), dtype=complex)
        expanded_indices = indices_func(
            n_electrons=particle_number, n_qubits=n_qubits, **kwargs
        )
        for iii in range(n_eigvecs):
            expanded_eigvecs[
                expanded_indices,
                iii,
            ] = eigvecs[:, iii]
        return eigvals, expanded_eigvecs
    return eigvals, eigvecs


def jw_get_true_ground_state_at_particle_number(
    sparse_operator: scipy.sparse.csc_matrix,
    particle_number: Union[list, int],
    spin: bool = True,
    sparse: bool = False,
    expanded: bool = True,
) -> Tuple[float, np.ndarray]:
    """Get the ground energy and wavefunction of a jw encoded fock hamiltonian
    Args:
        sparse_operator (scipy.sparse.csc_matrix): the sparse representation of the jw-encoded hamiltonian
        particle_number (Union[list,int]): the number of particle in the system. Can be a list of two items with the first being the up-spin particles and the second the down-spin ones
        spin (bool, optional): Whether to consider spin when looking for the ground state. Defaults to True.
        sparse (bool, optional): whether to use a sparse solver. Defaults to False.
    Returns:
        Tuple[float,np.ndarray]: the ground energy and the ground wavefunction
    """
    eigvals, eigvecs = jw_eigenspectrum_at_particle_number(
        sparse_operator,
        particle_number,
        expanded=expanded,
        spin=spin,
        sparse=sparse,
        k=1,
    )
    state = eigvecs[:, 0]
    return eigvals[0], state


def bravyi_kitaev_fast_wrapper(
    fermionic_operator: of.FermionOperator,
) -> of.QubitOperator:
    """Openfermion Bravyi-Kitaev wrapper for use in internal classes
    Args:
        fermionic_operator (of.FermionOperator): the fermion operator
    Returns:
        of.QubitOperator: the qubit operator
    """
    interaction_op = of.get_interaction_operator(fermionic_operator)
    return of.bravyi_kitaev_fast(interaction_op)


def mean_coeff_n_terms(fop: of.FermionOperator, n: int) -> float:
    """Get the mean coefficient of all fock terms of a certain order
    Args:
        fop (of.FermionOperator): the fock operators (containing multiple terms)
        n (int): the term order to take into account
    Returns:
        float: the mean of coefficients
    """
    mean_coeff = np.mean(list(fop.terms[k] for k in fop.terms.keys() if len(k) == n))
    return mean_coeff


def jw_computational_wf(
    indices: list, n_qubits: int, right_to_left: bool = False, expanded: bool = True
) -> np.ndarray:
    """Creates a 2^n_qubits wavefunction corresponding to the computational state of n_qubits with the qubits in indices set to one
    Args:
        indices (list): the indices of the qubits which are non-zero
        n_qubits (int): the number of qubits in the wavefunction
    Returns:
        np.ndarray: 2^n_qubits vector with a one in the entry correspknding to the desired computational state
    """

    wf = np.zeros((2**n_qubits), dtype=np.complex64)
    if right_to_left:
        jw_index = sum((2 ** (n_qubits - 1 - iii) for iii in indices))
    else:
        jw_index = sum((2 ** (iii) for iii in indices))

    wf[jw_index] = 1
    if not expanded:
        wf = wf[
            jw_spin_correct_indices(
                n_qubits=n_qubits,
                n_electrons=[sum_even(indices), sum_odd(indices)],
                right_to_left=right_to_left,
            )
        ]

    return wf


def is_subspace_gs_global(model, n_electrons: list):
    sys_eigenspectrum, _ = jw_eigenspectrum_at_particle_number(
        sparse_operator=of.get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    total_eigenspectrum, _ = np.linalg.eigh(
        model.hamiltonian.matrix(qubits=model.qubits)
    )
    print("subspace ground energy:", min(sys_eigenspectrum))
    print("global ground energy", min(total_eigenspectrum))
    print(
        "are they equal: ", np.isclose(min(sys_eigenspectrum), min(total_eigenspectrum))
    )


def ket_in_subspace(ket: np.ndarray, n_electrons: list, n_subspace_qubits: int):
    # check whether the current ket has a subsection of the bitstring in the right fermionic subspace
    # if your ket is tensored with an ancilla, you don't care about the state of some of the bitstring
    n_qubits = int(np.log2(len(ket)))
    n_non_subspace_qubits = n_qubits - n_subspace_qubits
    indices = []
    for ind in range(len(ket)):
        # turn the indices into rtl bitstrings, cut the non-subspace part, and count the ones
        subspace_qubits_indices = index_bits(
            binleftpad(ind, n_qubits)[n_non_subspace_qubits:],
            right_to_left=True,
        )
        if (
            sum_even(subspace_qubits_indices) == n_electrons[0]
            and sum_odd(subspace_qubits_indices) == n_electrons[1]
        ):
            # we got our subspace indices
            indices.append(ind)
        # look at the complementary other indices
        non_subspace_indices = np.array(list(set(range(len(ket))) - set(indices)))
    # if any of the compl. indices is not zero, then the ket is not in the subspace and return false
    return not np.any(ket[non_subspace_indices] != 0)


def get_bogo_fop(matrix: np.ndarray, spin: int):
    bogos = []
    for m in range(matrix.shape[0]):
        fop = of.FermionOperator()
        for n in range(matrix.shape[1]):
            fop += of.FermionOperator(f"{int(2*n + spin)}^", coefficient=matrix[m, n])
        bogos.append(fop)
    return bogos


def bogoprod(bogos: tuple[of.FermionOperator]):
    fop_out = of.FermionOperator.identity()
    for bogo in bogos:
        fop_out *= bogo
    return fop_out


def build_bogo_creators(
    bogos_up: list, bogos_down: list, n_electrons: list
) -> list[of.FermionOperator]:
    bogo_up_combs = list(itertools.combinations(bogos_up, n_electrons[0]))
    bogo_down_combs = list(itertools.combinations(bogos_down, n_electrons[1]))
    bogo_creators = []
    for i1, bogo_up_comb in enumerate(bogo_up_combs):
        for i2, bogo_down_comb in enumerate(bogo_down_combs):
            bogo_list = [*bogo_up_comb, *bogo_down_comb]
            bogo_creator = bogoprod(bogo_list)
            if bogo_creator is not None:
                bogo_creators.append(bogo_creator)

    return bogo_creators


def build_couplers_from_bogos(
    bogo_creators: list, zero_index: int = 0, max_k: int = -1
):
    bogo_0 = bogo_creators[zero_index]
    rest_bogos = [b for b in bogo_creators if b != bogo_0][:max_k]
    for bogo_c in rest_bogos:
        yield bogo_0 * of.hermitian_conjugated(bogo_c)


def get_bogoliubov_matrix(fop: of.FermionOperator, n_qubits: int):

    matrix = np.zeros((n_qubits, n_qubits))

    for k, v in fop.terms.items():
        coords = (k[0][0], k[1][0])
        matrix[coords] = v
    return matrix


def jw_get_free_couplers(
    model: "FermionicModel",
    zero_index: int = 0,
    max_k: int = -1,
    add_hc: bool = False,
    sort: bool = False,
) -> list[PauliSum]:
    """This creates the PauliSum corresponding to the Slater/free projectors/couplers: |psi_0><psi_k|

    Args:
        model (FermionicModel): the model from which to get the eigenstates
        zero_index (int, optional): which eigenstate is the ground state. Defaults to 0.
        max_k (int, optional): how many projector we should get. Defaults to -1.
        add_hc (bool, optional): add the hermitian conjugate of the coupler (+ |psi_k><psi_0|) . Defaults to False.

    Returns:
        list[PauliSum]: the Slater couplers
    """
    fop = of.get_fermion_operator(model.quadratic_terms)
    matrix = get_bogoliubov_matrix(fop, model.n_qubits)

    sector_matrix_up = matrix[::2, ::2]
    _, eigvecs_up = np.linalg.eigh(sector_matrix_up)
    sector_matrix_down = matrix[1::2, 1::2]
    _, eigvecs_down = np.linalg.eigh(sector_matrix_down)

    bogos_up = get_bogo_fop(eigvecs_up.T, 0)
    bogos_down = get_bogo_fop(eigvecs_down.T, 1)

    bogo_creators = build_bogo_creators(bogos_up, bogos_down, model.n_electrons)

    couplers = build_couplers_from_bogos(bogo_creators, zero_index, max_k)
    jw_couplers = []
    for coupler in couplers:
        jw_coupler = of.normal_ordered(coupler)
        jw_coupler = of.jordan_wigner(coupler)
        if add_hc:
            jw_coupler += of.hermitian_conjugated(jw_coupler)

        jw_couplers.append(
            of.qubit_operator_to_pauli_sum(jw_coupler, qubits=model.qubits)
        )

    return jw_couplers


def jw_hartree_fock_state(
    model: "FermionicModel",
    offset: int = 0,
    right_to_left: bool = True,
    expanded: bool = True,
):
    indices = [
        (2 * (i + offset)) % (model.n_qubits - 1) for i in range(model.n_electrons[0])
    ] + [
        (2 * (i + offset) + 1) % (model.n_qubits - 1)
        for i in range(model.n_electrons[1])
    ]
    return jw_computational_wf(
        indices=indices,
        n_qubits=model.n_qubits,
        right_to_left=right_to_left,
        expanded=expanded,
    )


def subspace_size(n_qubits: int, n_electrons: list) -> int:
    return len(
        list(itertools.combinations(range(n_qubits // 2), n_electrons[0]))
    ) * len(list(itertools.combinations(range(n_qubits // 2), n_electrons[1])))
