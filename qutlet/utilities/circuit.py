import itertools
from typing import TYPE_CHECKING, Tuple, Union

import cirq
import numpy as np
import sympy
from scipy.linalg import sqrtm
from scipy.sparse import csc_matrix, kron

from qutlet.utilities.generic import (
    chained_matrix_multiplication,
    default_value_handler,
    flatten,
)

if TYPE_CHECKING:
    from qutlet.models import QubitModel


# courtesy of cirq
def optimize_circuit(circuit, context=None, k=2):
    # Merge 2-qubit connected components into circuit operations.
    optimized_circuit = cirq.merge_k_qubit_unitaries(
        circuit, k=k, rewriter=lambda op: op.with_tags("merged"), context=context
    )

    # Drop operations with negligible effect / close to identity.
    optimized_circuit = cirq.drop_negligible_operations(
        optimized_circuit, context=context
    )

    # Expand all remaining merged connected components.
    optimized_circuit = cirq.expand_composite(
        optimized_circuit, no_decomp=lambda op: "merged" not in op.tags, context=context
    )

    # Synchronize terminal measurements to be in the same moment.
    optimized_circuit = cirq.synchronize_terminal_measurements(
        optimized_circuit, context=context
    )

    # Assert the original and optimized circuit are equivalent.
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, optimized_circuit
    )

    return optimized_circuit


def replace_symbols(circuit: cirq.Circuit, param_resover: cirq.ParamResolver):
    if param_resover is None:
        return circuit
    resolved_circuit = cirq.resolve_parameters(
        val=circuit, param_resolver=param_resover
    )
    return resolved_circuit


def cirq_circuit_to_target_gateset(
    circuit: cirq.Circuit,
    symbols: list[sympy.Symbol],
    params: np.ndarray,
    basis_gates: list,
):
    # resolve params
    if symbols is not None:
        param_resover = cirq.ParamResolver({s: v for s, v in zip(symbols, params)})
        resolved_circuit = replace_symbols(circuit, param_resover)
    else:
        resolved_circuit = circuit
    # optimize for good measure
    optimized_circuit = optimize_circuit(resolved_circuit)

    # finally convert
    transformed_circuit = cirq.optimize_for_target_gateset(
        circuit=optimized_circuit, gateset=basis_gates
    )
    return transformed_circuit


def count_n_qubit_gates(circuit: cirq.Circuit, max_locality: int = None):

    if max_locality is None:
        # max locality is at most the number of qubits
        max_locality = len(circuit.all_qubits())
    counts = np.zeros(max_locality)
    for ind in range(len(counts)):
        counts[ind] = count_n_qubit_gates(circuit, max_locality=ind + 1)
    return counts


def print_n_qubit_gates(circuit: cirq.Circuit):
    counts = count_n_qubit_gates(circuit)
    print(
        list(
            "{}-qubit gates: {}".format(ind + 1, val)
            for ind, val in enumerate(counts)
            if val != 0
        )
    )


def populate_empty_qubits(
    circuit: cirq.Circuit, qubits: list[cirq.Qid]
) -> cirq.Circuit:
    """Add I gates to qubits without operations. This is mainly to avoid some errors with measurement in cirq
    Args:
        circuit (cirq.Circuit): the circuit to check
    Returns:
        cirq.Circuit: the circuit with additional I gates
    """
    return identity_on_qubits(circuit=circuit, qubits=qubits)


def qmap(model: "QubitModel", reversed: bool = False) -> dict:
    """Get a qmap necessary for some openfermion functions
    Args:
        circuit (cirq.Circuit): the circuit we will use to generate the qmap
    Returns:
        dict: the resulting qmap
    """
    qubit_numbers = list(range(len(model.qubits)))
    if reversed:
        qubit_numbers = qubit_numbers[::-1]

    return {k: v for k, v in zip(model.qubits, qubit_numbers)}


def get_param_resolver(symbols: list, params: np.ndarray) -> cirq.ParamResolver:
    """Get a param resolver for cirq, i.e. put some numerical values in some symbolic items
    Args:
        symbols (list): the list of symbols
        params (np.ndarray, optional): the values to put in the place of the symbols.
    Returns:
        cirq.ParamResolver: the param resolver
    """
    joined_dict = {**{str(symbols[i]): params[i] for i in range(len(symbols))}}
    return cirq.ParamResolver(joined_dict)


def match_param_values_to_symbols(
    params: list, symbols: list, default_value: str = "zeros"
):
    """add values to param_values when some are missing wrt. the param array
    Args:
        params (params): the params are to be checked
        symbols (list): the symbols to match
        default_value (str, optional): what to put in the additional params. Defaults to "zeros".
    """
    if params is None:
        params = np.array([])
    missing_size = np.size(symbols) - np.size(params)

    if missing_size == 0:
        return

    elif missing_size < 0:
        for _ in range(abs(missing_size)):
            symbols.append(sympy.Symbol(name=f"msym_{len(symbols)}"))

    if missing_size > 0:

        param_default_values = default_value_handler(
            shape=(missing_size,), value=default_value
        )
        params = np.concatenate((params, param_default_values))


def sparse_pauli_string(pauli_str: Union[cirq.PauliString, str]):
    pauli_I = csc_matrix(np.array([[1, 0], [0, 1]]))
    pauli_X = csc_matrix(np.array([[0, 1], [1, 0]]))
    pauli_Y = csc_matrix(np.array([[0, -1j], [1j, 0]]))
    pauli_Z = csc_matrix(np.array([[1, 0], [0, -1]]))
    out = csc_matrix([1])
    d = {"I": pauli_I, "X": pauli_X, "Y": pauli_Y, "Z": pauli_Z}
    if isinstance(pauli_str, cirq.PauliString):
        for qid, pauli in reversed(pauli_str.items()):
            out = kron(d[str(pauli)], out)
    # if we have something like "XYZXZXZYX"
    elif isinstance(pauli_str, str):
        for pauli in reversed(pauli_str):
            out = kron(d[pauli], out)
    return out


def sparse_pauli_string_coeff(mat: csc_matrix, pauli_str: Union[cirq.PauliString, str]):
    pauli_sparse_matrix = sparse_pauli_string(pauli_str=pauli_str)
    product = mat @ pauli_sparse_matrix

    return product.trace() / mat.shape[0]


def pauli_string_coeff(
    mat: np.ndarray, pauli_product: list[cirq.Pauli], qubits: list[cirq.Qid]
):
    pauli_string = cirq.PauliString(*[m(q) for m, q in zip(pauli_product, qubits)])
    pauli_matrix = pauli_string.matrix(qubits=qubits)
    coeff = np.trace(mat @ pauli_matrix) / mat.shape[0]
    return coeff, pauli_string


def matrix_to_paulisum(
    mat: np.ndarray,
    qubits: list[cirq.Qid] = None,
    verbose: bool = False,
) -> cirq.PauliSum:
    if len(list(set(mat.shape))) != 1:
        raise ValueError("the matrix is not square")
    n_qubits = int(np.log2(mat.shape[0]))
    if qubits is None:
        qubits = cirq.LineQubit.range(n_qubits)
    pauli_matrices = ("I", "X", "Y", "Z")
    pauli_products = itertools.product(pauli_matrices, repeat=n_qubits)
    pauli_sum = cirq.PauliSum()
    for pauli_product in pauli_products:
        pauli_string = "".join(pauli_product)
        coeff = sparse_pauli_string_coeff(mat=csc_matrix(mat), pauli_str=pauli_string)
        if verbose:
            print(pauli_string, coeff, end="\r")
        if not np.isclose(np.abs(coeff), 0):
            pauli_sum += cirq.DensePauliString(
                pauli_mask=pauli_string, coefficient=coeff
            ).on(*qubits)
    return pauli_sum


def fidelity_wrapper(a, b, qid_shape=None, subspace_simulation: bool = False):
    if subspace_simulation:
        return fidelity(a, b)
    else:
        return cirq.fidelity(a, b, qid_shape=qid_shape)


def fidelity(a: np.ndarray, b: np.ndarray, use_eigvals: bool = False) -> float:
    """Returns the quantum fidelity between two objects, each of with being either a wavefunction (a vector) or a density matrix
    Args:
        a (np.ndarray): the first object
        b (np.ndarray): the second object
    Raises:
        ValueError: if there is a mismatch in the dimensions of the objects
        ValueError: if a tensor has more than 2 dimensions
    Returns:
        float: the fidelity between the two objects
    """
    # remove empty dimensions
    squa = np.squeeze(a)
    squb = np.squeeze(b)
    # check for dimensions mismatch
    if len(set((*squa.shape, *squb.shape))) > 1:
        raise ValueError("Dimension mismatch: {} and {}".format(squa.shape, squb.shape))
    # case for two vectors
    if len(squa.shape) == 1 and len(squb.shape) == 1:
        return np.real(
            np.sqrt(np.abs(np.dot(np.conj(squa), squb) * np.dot(np.conj(squb), squa)))
            ** 2
        )
    else:
        # case for one matrix and one vector, or two matrices
        items = []
        for item in (squa, squb):
            if len(item.shape) == 1:
                items.append(np.outer(item, item))
            elif len(item.shape) == 2:
                items.append(item)
            else:
                raise ValueError(
                    "expected vector or matrix, got {}dimensions".format(item.shape)
                )

        if use_eigvals:
            raise NotImplementedError("This part doesn't work yet")
            eigvals_a = np.linalg.eigvalsh(items[0])
            eigvals_b = np.linalg.eigvalsh(items[1])
            sqrt_eigvals_a = np.sqrt(eigvals_a + 0j)
            rho_sigma_rho = sqrt_eigvals_a * eigvals_b * sqrt_eigvals_a
            return np.real(np.sum(np.sqrt(rho_sigma_rho)) ** 2)
        else:
            items[0] = sqrtm(np.round(items[0], 10) + 0j)
            rho_sigma_rho = chained_matrix_multiplication(
                np.matmul, items[0], np.round(items[1], 10), items[0]
            )
            final_mat = sqrtm(rho_sigma_rho + 0j)
            return np.real(np.trace(final_mat) ** 2)


def state_fidelity_to_eigenstates(
    state: np.ndarray, eigenstates: np.ndarray, expanded: bool = True
):
    # eigenstates have shape N * M where M is the number of eigenstates

    fids = []

    for jj in range(eigenstates.shape[1]):
        if expanded:
            fids.append(
                cirq.fidelity(
                    state, eigenstates[:, jj], qid_shape=(2,) * int(np.log2(len(state)))
                )
            )
        else:
            # in case we have fermionic vectors which aren't 2**n
            # expanded refers to jw_ restricted spaces functions
            fids.append(fidelity(state, eigenstates[:, jj]) ** 2)
    return fids


def get_closest_degenerate_ground_state(
    ref_state: np.ndarray,
    comp_energies: np.ndarray,
    comp_states: np.ndarray,
):
    ix = np.argsort(comp_energies)
    comp_states = comp_states[:, ix]
    comp_energies = comp_energies[ix]
    comp_ground_energy = comp_energies[0]
    degeneracy = sum(
        (np.isclose(comp_ground_energy, eigenenergy) for eigenenergy in comp_energies)
    )
    fidelities = []
    if degeneracy > 1:
        print("ground state is {}-fold degenerate".format(degeneracy))
        for ind in range(degeneracy):
            fidelities.append(cirq.fidelity(comp_states[:, ind], ref_state))
        max_ind = np.argmax(fidelities)
        print(f"degenerate fidelities: {fidelities}, max: {max_ind}")
        return comp_ground_energy, comp_states[:, max_ind], max_ind
    else:
        return comp_ground_energy, comp_states[:, 0], 0


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
        raise TypeError("expected PauliString, got: {}".format(type(pstr)))
    return all(
        pstr.gate.pauli_mask
        == np.array([0] * len(pstr.gate.pauli_mask)).astype(np.uint8)
    )


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


def make_pauli_str_hermitian(
    pstr: cirq.PauliString, anti: bool = False
) -> cirq.PauliString:
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
        psum (cirq.PauliSum): the pauli sum input
        anti (bool, optional): whether to make it anti-hermitian. Defaults to False.
    Returns:
        cirq.PauliSum: the hermitian/anti-hermitian pauli sum
    """
    psum_out = sum([make_pauli_str_hermitian(pstr, anti) for pstr in psum])
    return psum_out


def pauli_sum_is_hermitian(psum: cirq.PauliSum, anti: bool = False):
    return all(pauli_str_is_hermitian(pstr=pstr, anti=anti) for pstr in psum)


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


def cRy(control, targ, sym: sympy.Symbol, fac: float = 1):
    return cirq.ControlledGate(
        sub_gate=cirq.Ry(rads=fac * np.pi * sym), num_controls=1
    ).on(control, targ)


def qnp_px(qubits: list[cirq.Qid] = None, sym: sympy.Symbol = None):
    # Anselmetti et al 2021 Nj. Phys 23 113010
    if qubits is None:
        qubits = cirq.LineQubit.range(4)
    if sym is None:
        sym = sympy.Symbol("theta")

    q0, q1, q2, q3 = qubits
    op_tree = []

    op_tree.append(cirq.CNOT(q1, q0))
    op_tree.append(cirq.CNOT(q3, q2))
    op_tree.append(cirq.CNOT(q3, q1))
    op_tree.append(cirq.X(q0))

    op_tree.append(cRy(q0, q3, fac=1 / 4, sym=sym))
    op_tree.append(cirq.CNOT(q0, q2))

    op_tree.append(cRy(q2, q3, fac=1 / 4, sym=sym))
    op_tree.append(cirq.CNOT(q0, q2))

    op_tree.append(cRy(q2, q3, fac=-1 / 4, sym=sym))
    op_tree.append(cirq.CZ(q1, q3))

    op_tree.append(cRy(q0, q3, fac=-1 / 4, sym=sym))
    op_tree.append(cirq.CNOT(q0, q2))

    op_tree.append(cirq.Rz(rads=np.pi / 2).on(q1))
    op_tree.append(cRy(q2, q3, fac=-1 / 4, sym=sym))

    op_tree.append(cirq.CNOT(q0, q2))

    op_tree.append(cirq.X(q0))
    op_tree.append(cRy(q2, q3, fac=1 / 4, sym=sym))

    op_tree.append(cirq.CNOT(q3, q1))
    op_tree.append(cirq.Rz(rads=-np.pi / 2).on(q1))
    op_tree.append(cirq.S(q3))

    op_tree.append(cirq.CNOT(q3, q2))
    op_tree.append(cirq.CNOT(q1, q0))

    return cirq.Circuit(*op_tree)


def identity_on_qubits(circuit: cirq.Circuit, qubits: list[cirq.Qid]) -> cirq.Circuit:
    """Adds I operations on all qubits from qubits which aren't on the circuit's qubits

    Args:
        circuit (cirq.Circuit): the circuit on which to add identities
        qubits (list[cirq.Qid]): the qubits to check against

    Returns:
        cirq.Circuit: the circuit with identities added
    """
    circuit_qubits = flatten(circuit.all_qubits())
    missing_qubits = [x for x in qubits if x not in circuit_qubits]
    circuit_out = circuit.copy()
    if circuit_qubits == []:
        print("The circuit has no qubits")
        circuit_out = cirq.Circuit()
    circuit_out.append([cirq.I(mq) for mq in missing_qubits])
    return circuit_out


def pretty_print_jw_order(
    pauli_string: cirq.PauliString, qubits: list[cirq.GridQubit]
):  # pragma: no cover
    last_qubit = max(qubits)
    mat = np.array(
        [["0" for y in range(last_qubit.col + 1)] for x in range(last_qubit.row + 1)]
    )
    for k, v in pauli_string.items():
        mat[(k.row, k.col)] = v
    mat = mat.tolist()
    print(pauli_string)
    print("\n".join(["".join(row) for row in mat]))


def print_state_fidelity_to_eigenstates(
    state: np.ndarray,
    eigenenergies: np.ndarray,
    eigenstates: np.ndarray,
    expanded: bool = True,
    decimals: int = 8,
):
    eig_fids = state_fidelity_to_eigenstates(
        state=state,
        eigenstates=eigenstates,
        expanded=expanded,
    )
    print("Populations")
    for ind, (fid, eigenenergy) in enumerate(zip(eig_fids, eigenenergies)):
        if not np.isclose(np.round(fid, decimals), 0):
            print(
                f"E_{ind:<{len(str(len(eigenenergies)))}}: fid: {np.abs(fid):.4f} gap: {np.abs(eigenenergy-eigenenergies[0]):.3f}"
            )
    print(f"sum fids {sum(eig_fids)}")
