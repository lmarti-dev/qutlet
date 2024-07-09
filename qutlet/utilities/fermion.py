import itertools
from typing import List, Tuple, Union
import numpy as np
import openfermion as of

import scipy
from qutlet.utilities.generic import sum_even, sum_odd, index_bits, binleftpad


def get_fermionic_states_number(system_fermions: list, n_qubits: int):
    return len(
        list(itertools.combinations(range(n_qubits // 2), system_fermions[0]))
    ) * len(list(itertools.combinations(range(n_qubits // 2), system_fermions[1])))


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


def double_excitation(
    coeff: float, i: int, j: int, k: int, l: int, anti_hermitian: bool
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
    system_fermions: Union[list, int],
    n_qubits: int,
    right_to_left: bool = False,
) -> list:
    """Get the indices corresponding to the spin sectors given by system_fermions
    Args:
        system_fermions (Union[list, int]): the system_fermions in the system, or the spin sectors
        n_qubits (int): the number of qubits in the system
    Raises:
        TypeError: if the system_fermions is neither a list nor an int
    Returns:
        list: the indices associated with the correct number of spins/electrons
    """
    # since we usually fill even then odd indices, and in the default jw,
    # the up spins are even and odd are down, we check if we have the correct
    # up and down spin count before passing the indices back
    # when system_fermions is odd, we assume there is one more even indices, since those are filled even-first.
    # I guess we could use abs() but then we would mix two unrelated parity states
    # in the case n_spin_up is defined, then we count the number of spin_up (even indices) in the comb

    if isinstance(system_fermions, int):
        # we get the correct combinations by checking there is the same amount of even and odd indices in the comb
        n_spin_up = int(np.ceil(system_fermions / 2))
        n_spin_down = int(np.floor(system_fermions / 2))
    elif isinstance(system_fermions, list):
        n_spin_up = system_fermions[0]
        n_spin_down = system_fermions[1]
    else:
        raise TypeError(
            "Expected system_fermions to be either a list or an int, got {}".format(
                type(system_fermions)
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
        system_fermions=particle_number, n_qubits=n_qubits, right_to_left=right_to_left
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
            system_fermions=particle_number, n_qubits=n_qubits, **kwargs
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
    indices: list, n_qubits: int, right_to_left: bool = False
) -> np.ndarray:
    """Creates a 2**Nqubits wavefunction corresponding to the computational state of Nqubits with the qubits in indices set to one
    Args:
        indices (list): the indices of the qubits which are non-zero
        Nqubits (int): the number of qubits in the wavefunction
    Returns:
        np.ndarray: 2**Nqubits vector with a one in the entry correspknding to the desired computational state
    """
    wf = np.zeros((2**n_qubits))
    if right_to_left:
        jw_index = sum((2 ** (n_qubits - 1 - iii) for iii in indices))
    else:
        jw_index = sum((2 ** (iii) for iii in indices))

    wf[jw_index] = 1
    return wf


def is_subspace_gs_global(model, Nf: list):
    sys_eigenspectrum, _ = jw_eigenspectrum_at_particle_number(
        sparse_operator=of.get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=Nf,
        expanded=True,
    )
    total_eigenspectrum, _ = np.linalg.eigh(
        model.hamiltonian.matrix(qubits=model.flattened_qubits)
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
