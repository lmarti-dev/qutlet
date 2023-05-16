import cirq
from typing import Tuple, Union
import itertools
import fauvqe.utils as utils
import numpy as np
import openfermion as of
from fauvqe.models.abstractmodel import AbstractModel
import io
from typing import List
import scipy


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


def qubits_shape(qubits: Tuple[cirq.Qid]) -> tuple:
    """Get the shape of some qubit list

    Args:
        qubits (Tuple[cirq.Qid]): the qubit list

    Returns:
        tuple: the shape of the qubits
    """
    last_qubit = max(qubits)
    if isinstance(last_qubit, cirq.LineQubit):
        return (last_qubit.x + 1, 1)
    elif isinstance(last_qubit, cirq.GridQubit):
        return (last_qubit.row + 1, last_qubit.col + 1)


# shamelessly taken from stack
def depth(circuit: cirq.Circuit) -> int:
    """Get the depth of a circuit
    We create a new circuit to repack it, since the original circuit whose depth we want to compute has not been optimised.
    

    Args:
        circuit (cirq.Circuit): the circuit to evaluate

    Returns:
        int: the depth of the circuit
    """
    depth = len(cirq.Circuit(circuit.all_operations()))
    return depth


def jw_spin_correct_indices(n_electrons: Union[list, int], n_qubits: int) -> list:
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
    # when Nf is odd, we assume there is one more even indices, since those are filled even-first.
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
            "Expected n_electrons to be either a list or an int, got {}".format(type(n_electrons))
        )
    combinations = itertools.combinations(range(n_qubits), sum([n_spin_up, n_spin_down]))
    correct_combinations = [
        list(c)
        for c in combinations
        if utils.sum_even(c) == n_spin_up and utils.sum_odd(c) == n_spin_down
    ]
    jw_indices = [sum([2**iii for iii in combination]) for combination in correct_combinations]
    return jw_indices


# these are modified openfermion functions: they used a non-deterministic way (scipy.sparse.eigh) to find the gs which messes up
# any sort of meaningful comparison (when the gs is degenerate). This allows for the slower, but deterministic dense numpy eig.
# also implemented is the possibility to further restrict the Fock space to match spin occupations, and uneven spin sectors


def jw_spin_restrict_operator(
    sparse_operator: scipy.sparse.csc_matrix, particle_number: Union[list, int], n_qubits: int
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

    select_indices = jw_spin_correct_indices(n_electrons=particle_number, n_qubits=n_qubits)
    return sparse_operator[np.ix_(select_indices, select_indices)]


# Again this function comes from openfermion, I changed the eigenvalue function to a deterministic one so that I get the same ground state everytime in case it is degenerate
def eigenspectrum_at_particle_number(
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
        Tuple[np.ndarray,np.ndarray]: _description_
    """

    n_qubits = int(np.log2(sparse_operator.shape[0]))
    # Get the operator restricted to the subspace of the desired particle number

    if spin:
        restricted_operator_func = jw_spin_restrict_operator
        indices_func = jw_spin_correct_indices
    else:
        # if you specified spin sectors but want to ignore spin, merge sectors
        if isinstance(particle_number, list):
            particle_number = sum(particle_number)
        restricted_operator_func = of.jw_number_restrict_operator
        indices_func = of.jw_number_indices
    restricted_operator = restricted_operator_func(sparse_operator, particle_number, n_qubits)
    # Compute eigenvalues and eigenvectors
    if sparse:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(restricted_operator, k=k, which="SA")
    else:
        dense_restricted_operator = restricted_operator.toarray()
        eigvals, eigvecs = np.linalg.eigh(dense_restricted_operator)
    if expanded:
        if k is not None and sparse:
            n_eigvecs = k
        else:
            n_eigvecs = 2**n_qubits
        expanded_eigvecs = np.zeros((2**n_qubits, n_eigvecs), dtype=complex)
        for iii in range(n_eigvecs):
            expanded_eigvecs[
                indices_func(n_electrons=particle_number, n_qubits=n_qubits),
                iii,
            ] = eigvecs[:, iii]
            return eigvals, expanded_eigvecs
    return eigvals, eigvecs


def jw_get_true_ground_state_at_particle_number(
    sparse_operator: scipy.sparse.csc_matrix,
    particle_number: Union[list, int],
    spin: bool = True,
    sparse: bool = False,
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
    eigvals, eigvecs = eigenspectrum_at_particle_number(
        sparse_operator,
        particle_number,
        expanded=True,
        spin=spin,
        sparse=sparse,
        k=1,
    )
    state = eigvecs[:, 0]
    return eigvals[0], state


def get_param_resolver(model: AbstractModel, param_values: np.ndarray) -> cirq.ParamResolver:
    """Get a param resolver for cirq, i.e. put some numerical values in some symbolic items

    Args:
        model (AbstractModel): the model for which we want a param
        param_values (np.ndarray, optional): the values to put in the place of the symbols.

    Returns:
        cirq.ParamResolver: the param resolver
    """
    joined_dict = {
        **{str(model.circuit_param[i]): param_values[i] for i in range(len(model.circuit_param))}
    }
    return cirq.ParamResolver(joined_dict)


def pauli_str_is_hermitian(pstr: cirq.PauliString, anti: bool = False) -> bool:
    """Check whether some Pauli string is hermitian or anti-hermitian

    Args:
        pstr (cirq.PauliString): the pauli string to check
        anti (bool, optional): Whether to check for hermitianity or anti-hermitianity. Defaults to False.

    Returns:
        bool: whether the pauli string is hermitian/anti-hermitian
    """
    if anti:
        return np.conj(pstr.coefficient) == -pstr.coefficient
    else:
        return np.conj(pstr.coefficient) == pstr.coefficient


def pauli_sum_is_hermitian(psum: cirq.PauliSum, anti: bool = False):
    return all(pauli_str_is_hermitian(pstr=pstr, anti=anti) for pstr in psum)


def make_pauli_str_hermitian(pstr: cirq.PauliString, anti: bool = False) -> cirq.PauliString:
    """Make a

    Args:
        pstr (cirq.PauliString): _description_
        anti (bool, optional): _description_. Defaults to False.

    Returns:
        cirq.PauliString: _description_
    """
    if pauli_str_is_hermitian(pstr=pstr, anti=anti):
        return pstr
    elif pauli_str_is_hermitian(pstr=pstr, anti=not anti):
        return pstr.with_coefficient(1j * pstr.coefficient)
    else:
        if not anti:
            # hermitian A + A* = re(A)
            return pstr.with_coefficient(np.real(pstr.coefficient))
        else:
            # anti-hermitian A - A* = 1j*im(A)
            return pstr.with_coefficient(1j * np.imag(pstr.coefficient))


def make_pauli_sum_hermitian(psum: cirq.PauliSum, anti: bool = False) -> cirq.PauliSum:
    """Make a pauli sum hermitian or anti-hermitian

    Args:
        psum (cirq.PauliSum): the pauli sum inpuer
        anti (bool, optional): whether to make it anti-hermitian. Defaults to False.

    Returns:
        cirq.PauliSum: the hermitian/anti-hermitian pauli sum
    """
    psum_out = sum([make_pauli_str_hermitian(pstr, anti) for pstr in psum])
    return psum_out


def qmap(model: AbstractModel) -> dict:
    """Get a qmap necessary for some openfermion functions

    Args:
        model (AbstractModel): the model we will use to generate the qmap

    Returns:
        dict: the resulting qmap
    """
    flattened_qubits = list(utils.flatten(model.qubits))
    return {k: v for k, v in zip(flattened_qubits, range(len(flattened_qubits)))}


def populate_empty_qubits(model: AbstractModel) -> cirq.Circuit:
    """Add I gates to qubits without operations. This is mainly to avoid some errors with measurement in cirq

    Args:
        model (AbstractModel): the model to check

    Returns:
        cirq.Circuit: the circuit with additional I gates
    """
    circuit_qubits = list(model.circuit.all_qubits())
    model_qubits = model.flattened_qubits
    missing_qubits = [x for x in model_qubits if x not in circuit_qubits]
    circ = model.circuit.copy()
    if circuit_qubits == []:
        print("The circuit has no qubits")

        circ = cirq.Circuit()
    circ.append([cirq.I(mq) for mq in missing_qubits])
    return circ


def match_param_values_to_symbols(
    model: AbstractModel, symbols: list, default_value: str = "zeros"
):
    """add values to param_values when some are missing wrt. the param array

    Args:
        model (AbstractModel): the model whose params are to be checked
        symbols (list): the symbols to match
        default_value (str, optional): what to put in the additional params. Defaults to "zeros".
    """
    if model.circuit_param_values is None:
        model.circuit_param_values = np.array([])
    missing_size = np.size(symbols) - np.size(model.circuit_param_values)

    param_default_values = utils.default_value_handler(shape=(missing_size,), value=default_value)
    if missing_size > 0:
        model.circuit_param_values = np.concatenate(
            (model.circuit_param_values, param_default_values)
        )


def pauli_str_is_identity(pstr: cirq.PauliString) -> bool:
    """Check whether a Pauli string is the identity, i.e. like IIIIIII

    Args:
        pstr (cirq.PauliString): the pauli string to check

    Raises:
        ValueError: if the input is not a Pauli string

    Returns:
        bool: whether it is the identity
    """
    if not isinstance(pstr, cirq.PauliString):
        raise ValueError("expected PauliString, got: {}".format(type(pstr)))
    return all(pstr.gate.pauli_mask == np.array([0] * len(pstr.gate.pauli_mask)).astype(np.uint8))


def all_pauli_str_commute(psum: cirq.PauliSum) -> bool:
    """check whether all terms in a PauliSum commute with one another

    Args:
        psum (cirq.PauliSum): the Pauli sum to check

    Returns:
        bool: whether they commute
    """

    for pstr1 in psum:
        for pstr2 in psum:
            if not cirq.commutes(pstr1, pstr2):
                return False
    return True


def even_excitation(coeff: float, indices: List[int], anti_hermitian: bool) -> of.FermionOperator:
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
            "expected an even length for the indices list but got: {}".format(len(indices))
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


def single_excitation(coeff: float, i: int, j: int, anti_hermitian: bool) -> of.FermionOperator:
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
    return even_excitation(coeff=coeff, indices=[i, j, k, l], anti_hermitian=anti_hermitian)


def bravyi_kitaev_fast_wrapper(fermionic_operator: of.FermionOperator) -> of.QubitOperator:
    """Openfermion Bravyi-Kitaev wrapper for use in internal classes

    Args:
        fermionic_operator (of.FermionOperator): the fermion operator

    Returns:
        of.QubitOperator: the qubit operator
    """
    interaction_op = of.get_interaction_operator(fermionic_operator)
    return of.bravyi_kitaev_fast(interaction_op)
