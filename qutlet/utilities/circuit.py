import itertools
import re
from typing import TYPE_CHECKING, Tuple, Union

import cirq
import cirq.circuits
import numpy as np
import sympy
from scipy.linalg import sqrtm
from scipy.sparse import csc_matrix, kron


from qutlet.utilities.generic import (
    chained_matrix_multiplication,
    default_value_handler,
    flatten,
)
from math import prod
from qutlet.utilities.fermion import jw_spin_correct_indices


if TYPE_CHECKING:
    from qutlet.models import QubitModel


def expectation_wrapper(
    observable: Union[cirq.PauliSum, np.ndarray],
    state: np.ndarray,
    qubits: list[cirq.Qid],
):
    if len(state.shape) == 2:
        if isinstance(observable, cirq.PauliSum):
            return np.real(
                observable.expectation_from_density_matrix(
                    state.astype("complex_"),
                    qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
                )
            )
        elif isinstance(observable, np.ndarray):
            return np.real(np.trace(observable @ state))
    else:
        if isinstance(observable, cirq.PauliSum):
            return np.real(
                observable.expectation_from_state_vector(
                    state.astype("complex_"),
                    qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
                )
            )
        elif isinstance(observable, np.ndarray):
            return np.real(np.vdot(state, observable @ state))
    raise ValueError(
        f"Got an incompatible observable and state: observable {type(observable)}, state: {type(state)}"
    )


def conjugate(val: Union[cirq.PauliSum, cirq.PauliString]):
    if isinstance(val, cirq.PauliString):
        coeff = val.coefficient
        return val.with_coefficient(np.real(coeff) - 1j * np.imag(coeff))
    elif isinstance(val, cirq.PauliSum):
        return cirq.PauliSum.from_pauli_strings([conjugate(pstr) for pstr in val])
    else:
        raise ValueError(f"Expected PauliString or PauliSum, got: {type(val)}")


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


def count_n_qubit_gates(circ: cirq.Circuit, n: int):
    n_qubit_gates = 0
    for moment in circ:
        for item in moment:
            if len(item.qubits) == n:
                n_qubit_gates += 1
    return n_qubit_gates


def count_qubit_gates_to_locality(circuit: cirq.Circuit, max_locality: int = None):

    if max_locality is None:
        # max locality is at most the number of qubits
        max_locality = len(circuit.all_qubits())
    counts = np.zeros(max_locality)
    for ind in range(len(counts)):
        counts[ind] = count_n_qubit_gates(circuit, n=ind + 1)
    return counts


def print_n_qubit_gates(circuit: cirq.Circuit):
    counts = count_qubit_gates_to_locality(circuit)
    for ind, val in enumerate(counts):
        if val != 0:
            print(f"{ind+1}-qubit gates: {val}")


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
    return params, symbols


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
        fids.append(
            fidelity_wrapper(
                state,
                eigenstates[:, jj],
                qid_shape=(2,) * int(np.log2(len(state))),
                subspace_simulation=not expanded,
            )
        )
    return fids


def get_closest_state(
    ref_state: np.ndarray, comp_states: np.ndarray, subspace_simulation: bool = False
) -> Tuple[np.ndarray, int]:
    fidelities = []
    for ind in range(comp_states.shape[1]):
        fid = fidelity_wrapper(
            comp_states[:, ind],
            ref_state,
            qid_shape=(2,) * int(np.log2(len(ref_state))),
            subspace_simulation=subspace_simulation,
        )
        if fid > 0.5:
            # if one state has more than .5 fid with ref, then it's necessarily the closest
            return comp_states[:, ind], int(ind)
        fidelities.append(fid)
    max_ind = np.argmax(fidelities)
    print(f"{len(fidelities)} degenerate fidelities. max: {max_ind}")
    return comp_states[:, max_ind], int(max_ind)


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
    else:
        return (len(qubits), 1)


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


def build_max_magic_state(n_qubits: int) -> np.ndarray:
    """Build a density matrix which is as little a stabilizer state as possible

    Args:
        n_qubits (int): number of qubit

    Returns:
        np.ndarray: the density matrix
    """
    paulis = ["I", "X", "Y", "Z"]
    pauli_strings = itertools.product(paulis, repeat=n_qubits)
    qubits = cirq.LineQubit.range(n_qubits)
    psum = cirq.PauliSum()
    dim = 2**n_qubits
    for pauli_string in pauli_strings:
        if pauli_string != "I" * n_qubits:
            psum += cirq.DensePauliString(pauli_string).on(*qubits) / (dim * (dim + 1))
        else:
            psum += cirq.DensePauliString(pauli_string).on(*qubits) / (dim)
    rho = psum.matrix(qubits=qubits)
    rho /= np.trace(rho)
    return rho


def hartree_fock_circuit(
    qubits: cirq.Qid, n_electrons: list, reversed: bool = False
) -> cirq.Circuit:
    circ = cirq.Circuit()
    n_qubits = len(qubits)
    for x in range(n_electrons[0]):
        if reversed:
            circ.append(cirq.X(qubits[n_qubits - 1 - 2 * x]))
        else:
            circ.append(cirq.X(qubits[2 * x]))
    for x in range(n_electrons[1]):
        if reversed:
            circ.append(cirq.X(qubits[n_qubits - 1 - (2 * x + 1)]))
        else:
            circ.append(cirq.X(qubits[2 * x + 1]))
    return circ


def prettify_pstr(pstr: cirq.PauliString, n_qubits: int):
    pm = "IXYZ"
    pd = pauli_dict(pstr)
    paulis = [pm[pd[j]] if j in pd.keys() else "I" for j in range(n_qubits)]
    c = pstr.coefficient
    if np.conj(c) == -c:
        c_str = f"{np.imag(c):.4f}i "
    elif np.conj(c) == c:
        c_str = f"{np.real(c):.4f} "
    else:
        c_str = f"({np.real(c):.4f}+{np.imag(c):.4f}i) "
    s = c_str + "".join(paulis)
    return s


def pauli_neighbour_order(pstr: cirq.PauliString):
    qs = pstr.qubits
    if not all(isinstance(q, cirq.LineQubit) for q in qs):
        raise ValueError(f"Expected LineQubit, got: {type(qs)}")
    if len(qs):
        return np.abs(max(qs).x - min(qs).x)
    return 0


def pauli_basis_change(pstr: cirq.PauliString) -> cirq.Circuit:
    circ = cirq.Circuit()
    pm = "IXYZ"
    pd = {q: v for q, v in zip(pstr.qubits, pstr.gate.pauli_mask)}
    for j in pd.keys():
        pauli = pm[pd[j]]
        if pauli == "I":
            ...
        elif pauli == "X":
            circ += cirq.H(j)
        elif pauli == "Y":
            circ += cirq.inverse(cirq.S(j))
            circ += cirq.H(j)
        elif pauli == "Z":
            ...
    pni = [q for q, v in zip(pstr.qubits, pstr.gate.pauli_mask) if v != 0]
    for j in range(len(pni) - 1):
        circ += cirq.CNOT(pni[j], pni[j + 1])

    return circ, pni[-1].x


def amplitude_amplification_projector(
    exclude_idx: np.ndarray,
    n_qubits: int,
    n_electrons: list,
    right_to_left: bool = False,
    perp: bool = True,
    subspace_simulation: bool = False,
) -> np.ndarray:

    #  Fixed-Point Adiabatic Quantum Search eq 4
    # nielsen chuang pg 70
    proj = np.zeros((2**n_qubits, 2**n_qubits))
    proj[exclude_idx, exclude_idx] = 1

    if perp:
        idx = jw_spin_correct_indices(
            n_electrons=n_electrons, n_qubits=n_qubits, right_to_left=right_to_left
        )
        id_mat = np.zeros((2**n_qubits, 2**n_qubits))
        id_mat[idx, idx] = 1
        out = id_mat - proj
    else:
        out = proj

    if subspace_simulation:
        if not perp:
            idx = jw_spin_correct_indices(
                n_electrons=n_electrons, n_qubits=n_qubits, right_to_left=right_to_left
            )
        return out[np.ix_(idx, idx)]
    return out


def bitstring_to_ham(
    bitstring: str, qubits: cirq.Qid, right_to_left: bool = False
) -> cirq.PauliSum:
    ham: cirq.PauliSum = cirq.PauliSum()
    if right_to_left:
        bitstring = bitstring[::-1]
    for ind, s in enumerate(bitstring):
        if int(s) == 1:
            ham += cirq.Z(qubits[ind])
        else:
            ham -= cirq.Z(qubits[ind])
    return ham


def get_diagonal_ham_terms(ham: cirq.PauliSum) -> cirq.PauliSum:
    out_ham = cirq.PauliSum()
    for pstr in ham:
        if re.fullmatch(r"[IZ]+", pstr_to_str(pstr, len(ham.qubits))):
            out_ham += pstr
    return out_ham


def pauli_dict(pstr: cirq.PauliString) -> dict:
    return {ind: v for ind, v in enumerate(pstr.gate.pauli_mask)}


def pstr_to_str(pstr: cirq.PauliString, n_qubits: int) -> str:
    pm = "IXYZ"
    pd = pauli_dict(pstr)
    return "".join([pm[pd[j]] if j in pd.keys() else "I" for j in range(n_qubits)])


def get_projector_psum(qubits: cirq.Qid, bitstring: str, right_to_left: bool = True):
    if right_to_left:
        bitstring = bitstring[::-1]

    b: cirq.PauliString = prod(
        [cirq.X(qubits[ind]) for ind in range(len(qubits)) if bitstring[ind] == "1"]
    )
    zero_zero = prod(
        [
            0.5 * (cirq.I(qubits[ind]) + cirq.Z(qubits[ind]))
            for ind in range(len(qubits))
        ]
    )

    return b * zero_zero * b


def psum_to_subspace_matrix(
    psum: cirq.PauliSum, n_electrons: list, qubits: list[cirq.Qid]
) -> np.ndarray:
    idx = jw_spin_correct_indices(
        n_electrons=n_electrons, n_qubits=len(qubits), right_to_left=True
    )
    return psum.matrix(qubits)[np.ix_(idx, idx)]


def mat_to_psum(
    mat: np.ndarray,
    qubits: list[cirq.Qid] = None,
    verbose: bool = False,
) -> cirq.PauliSum:
    if len(list(set(mat.shape))) != 1:
        raise ValueError("the matrix is not square")
    n_qubits = int(np.log2(mat.shape[0]))
    if qubits is None:
        qubits = cirq.LineQubit.range(n_qubits)
    pauli_matrices = (cirq.I, cirq.X, cirq.Y, cirq.Z)
    pauli_products = itertools.product(pauli_matrices, repeat=n_qubits)
    pauli_sum = cirq.PauliSum()
    for pauli_product in pauli_products:
        coeff, pauli_string = pauli_string_coeff(mat, pauli_product, qubits)
        if verbose:
            print(pauli_product, coeff)
        if not np.isclose(np.abs(coeff), 0):
            pauli_sum += cirq.PauliString(pauli_string, coeff)
    return pauli_sum
