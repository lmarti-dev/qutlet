import numpy as np

from openfermion import expectation, get_sparse_operator, FermionOperator
from cirq import partial_trace, fidelity, DensePauliString, PauliString, LineQubit
from qutlet.utilities.fermion import (
    jw_eigenspectrum_at_particle_number,
    index_bits,
)

from qutlet.utilities.generic import (
    from_bitstring,
    to_bitstring,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qutlet.models import FermionicModel

import itertools


def stabilizer_renyi_entropy(
    state: np.ndarray, n_qubits: int, alpha: float = 2, normalize: bool = True
) -> float:
    # 1. Leone, L., Oliviero, S. F. E. & Hamma, A. Stabilizer R\’enyi Entropy. Phys. Rev. Lett. 128, 050402 (2022).
    # if normalize is true, the stabilizer Rényi entropy is upper bounded by 1. (otherwise log(2**n))

    pauli_strings: list[PauliString] = []
    paulis = ["I", "X", "Y", "Z"]
    pauli_strings = itertools.product(paulis, repeat=n_qubits)
    expectation_values = []
    dimension = 2**n_qubits
    qubits = LineQubit.range(n_qubits)
    for pauli_string in pauli_strings:
        pstr = DensePauliString(pauli_string).on(*qubits)
        expectation_value = (
            pstr.expectation_from_state_vector(
                state, qubit_map={v: k for k, v in enumerate(qubits)}
            )
            ** 2
        )

        expectation_value /= dimension
        expectation_values.append(expectation_value)
    entropy = np.real(
        (alpha / (1 - alpha)) * np.log(np.linalg.norm(expectation_values, ord=alpha))
        - np.log(dimension)
    )
    if normalize:
        entropy /= np.log(dimension)
    return entropy


def correlation_entropy(state: np.ndarray, n_electrons: list, n_qubits: int):
    # Complexity of fermionic states. Preprint at https://doi.org/10.48550/arXiv.2306.07584 (2023).
    corr_mat = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(n_qubits):
            fop = FermionOperator(f"{j}^ {i}")
            corr_mat[i, j] = np.real(
                expectation(get_sparse_operator(fop, n_qubits=n_qubits), state)
            )
    electron_eigvals, _ = np.linalg.eigh(corr_mat)
    hole_eigvals = 1 - electron_eigvals
    electron_entropy = -np.log(np.sum(electron_eigvals**2) / np.sum(n_electrons))
    hole_entropy = -np.log(np.sum(hole_eigvals**2) / (n_qubits - np.sum(n_electrons)))
    return np.max([hole_entropy, electron_entropy])


def all_possible_bipartites(state: np.ndarray, n_qubits: int):
    bipartites = []
    for len_size_a in range(1, n_qubits):
        combs = itertools.combinations(range(n_qubits), len_size_a)
        for idx in combs:
            bipartites.append(
                bipartite_entanglement_entropy(state=state, n_qubits=n_qubits, idx=idx)
            )
    return bipartites


def is_invalid(A: np.ndarray):
    return np.any(np.isnan(A)) or np.any(np.isinf(A))


def concurrence(rho: np.ndarray, n_qubits: int, idx: int):
    rho_m = partial_trace(
        rho.reshape(*[2 for _ in range(2 * n_qubits)]),
        idx,
    ).reshape(2, 2)
    return np.sqrt(2 * (1 - np.trace(rho_m**2)))


def global_entanglement(state: np.ndarray, n_qubits: int):
    rho = np.outer(state.T.conjugate(), state)
    concurrences = np.zeros(n_qubits)
    for idx in range(n_qubits):
        concurrences[idx] = concurrence(rho=rho, n_qubits=n_qubits, idx=idx)
    return np.sum(concurrences)


def int_to_int_via_bitstring_trunc(i: int, j: int, n_qubits: int, right_to_left: bool):
    # turn an int i to a bitstring, remove the jth bit and back to int
    bs = to_bitstring(i, n_qubits=n_qubits, right_to_left=right_to_left)
    nbs = bs[:j] + bs[j + 1 :]
    out_i = from_bitstring(nbs, n_qubits=n_qubits, right_to_left=right_to_left)
    return out_i


def inodotj(j: int, b: bool, state: np.ndarray, right_to_left: bool = False):
    n_qubits = int(np.log2(state.shape[0]))
    out_state = np.zeros(2 ** int(n_qubits - 1))
    for ind, coeff in enumerate(state):
        ones_idx = index_bits(
            a=ind, N=n_qubits, right_to_left=right_to_left, ones=bool(b)
        )

        if j in ones_idx:
            c = 1
        else:
            c = 0
        ind_out = int_to_int_via_bitstring_trunc(
            i=ind, j=j, n_qubits=n_qubits, right_to_left=right_to_left
        )
        out_state[ind_out] += np.real(c * coeff)
    # print(j, b, out_state)
    return out_state


def wedge_squared(u: np.ndarray, v: np.ndarray):
    out = 0
    for y in range(len(u)):
        for x in range(y):
            out += (u[x] * v[y] - u[y] * v[x]) ** 2

    return out


def global_entanglement_wallach(state: np.ndarray, n_qubits: int):
    # Global entanglement in multiparticle systems. Journal of Mathematical Physics 43, 4273–4278 (2002).
    out = 0
    for n in range(n_qubits):
        out += wedge_squared(
            inodotj(j=n, b=0, state=state), inodotj(j=n, b=1, state=state)
        )
    return 4 * out / n_qubits


def bipartite_entanglement_entropy(state: np.ndarray, n_qubits: int, idx: list = None):
    if idx is None:
        idx = range(0, n_qubits, 2)
    rho = np.outer(state.T.conjugate(), state)
    rho_a = partial_trace(
        rho.reshape(*[2 for _ in range(2 * n_qubits)]),
        idx,
    ).reshape(int(2 ** len(idx)), int(2 ** len(idx)))
    eigvals, _ = np.linalg.eigh(rho_a)

    # some values are like -1e-40 fudging the entropy to nan when it should be 0
    # so we only use the values which are above 0
    # lim x->0 xlogx is 0 anyway
    return -np.real(eigvals @ np.log(eigvals, where=eigvals > 0))


def max_slater_fidelity(model: "FermionicModel"):
    # M. Optimal multi-configuration approximation of an N-fermion wave function. Phys. Rev. A 89, 012504 (2014).
    free_energies, free_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.get_quadratic_hamiltonian_wrapper(model.fock_hamiltonian)
        ),
        particle_number=model.n_electrons,
        expanded=True,
    )
    fidelities = np.zeros((len(free_energies)))
    ground_energy, ground_state = model.gs
    for ind in range(free_states.shape[-1]):
        fidelities[ind] = fidelity(
            ground_state, free_states[:, ind], qid_shape=(2,) * model.n_qubits
        )
    return np.max(fidelities)


def molecular_complexity(geom):
    # https://www.nature.com/articles/s41598-018-37253-8
    geom = None  # noqa: F841
    raise NotImplementedError("Haven't done this one yet (you can do it now)")


def quantum_wasserstein_distance(rho: np.ndarray, sigma: np.ndarray):
    # 1. Li, L., Bu, K., Koh, D. E., Jaffe, A. & Lloyd, S. Wasserstein Complexity of Quantum Circuits. Preprint at https://doi.org/10.48550/arXiv.2208.06306 (2022).
    raise NotImplementedError("Haven't done this one yet either (you can do it now)")
