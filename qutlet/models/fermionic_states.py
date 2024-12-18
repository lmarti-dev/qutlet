from typing import Sequence, Union

import cirq
import openfermion as of

from qutlet.models.fermionic_model import FermionicModel
from qutlet.utilities import (
    index_bits,
    sum_even,
    sum_odd,
)


def set_initial_state_circuit(
    circuit: cirq.Circuit,
    name: str,
    initial_state: Union[int, Sequence[int]] = None,
):
    """Inserts the cirq.OP_TREE generated by _get_initial_state_circuit into the circuit

    Args:
        name (str): the name of the type of initial state desired
        n_electrons: the number of fermions in the system
        initial_state (Union[int, Sequence[int]], optional): the indices of qubits that start n the 1 state. Defaults to 0 (i.e. all flipped down).
        An int input will be converted to binary and interpreted as a computational basis vector
        e.g. 34 = 100010 means the first and fifth qubits are initialized at one.
        rows (int): the rows taken from the Q matrix (rows of Q), where Q is defined from b* = Qa*, with a* creation operators.
                                                            Q diagonalizes n_electrons rows of the non-interacting hamiltonian
    """
    initial_state = process_initial_state_input(initial_state=initial_state)
    op_tree = get_initial_state_circuit(name=name, initial_state=initial_state)
    if op_tree is not None:
        circuit.append(op_tree)


def process_initial_state_input(
    model: FermionicModel,
    initial_state: Union[int, Sequence[int]] = None,
):
    # this method exists because the way to initiate the circuit is not so straight-forward
    # one could either put a int in initial state to get a binary computational state
    # or use nh to get some spin sectors. It's all very complex and deserves its own function
    if isinstance(initial_state, int):
        initial_state = index_bits(bin(initial_state), right_to_left=True)
    if initial_state is None:
        initial_state = list(
            sorted(
                [2 * k for k in range(model.n_electrons[0])]
                + [2 * k + 1 for k in range(model.n_electrons[1])]
            )
        )
    if (
        len(initial_state) != sum(model.n_electrons)
        or sum_even(initial_state) != model.n_electrons[0]
        or sum_odd(initial_state) != model.n_electrons[1]
    ):
        raise ValueError(
            "Mismatch between initial state and desired number of fermions. Initial state: {}, n_electrons: {}".format(
                initial_state, model.n_electrons
            )
        )
    return initial_state


def gaussian_state_circuit(model: FermionicModel):
    quadratic_hamiltonian = model.get_quadratic_hamiltonian_wrapper(
        model.fock_hamiltonian
    )
    op_tree = of.prepare_gaussian_state(
        qubits=model.qubits,
        quadratic_hamiltonian=quadratic_hamiltonian,
        occupied_orbitals=list(range(sum(model.n_electrons))),
        initial_state=0,
    )
    return op_tree


def slater_state_circuit(model: FermionicModel):
    _, unitary_rows = model.diagonalize_non_interacting_hamiltonian()
    op_tree = of.prepare_slater_determinant(
        qubits=model.qubits,
        slater_determinant_matrix=unitary_rows[: sum(model.n_electrons), :],
        initial_state=0,
    )
    return op_tree


def computational_state_circuit(initial_state, qubits):
    op_tree = [cirq.X(qubits[ind]) for ind in initial_state]
    return op_tree


def uniform_superposition_state_circuit(initial_state, qubits):
    op_tree = computational_state_circuit(initial_state, qubits)
    op_tree += [cirq.H(qubits[ind]) for ind in initial_state]
    return op_tree


def dicke_state_circuit(n_electrons, qubits):
    # TODO: implement the circuit
    op_tree = []
    return op_tree


def get_initial_state_circuit(
    name: str, initial_state: Union[int, Sequence[int]], model: FermionicModel
):
    if name is None:
        name == "none"

    if name == "none":
        return None
    elif name == "computational":
        return computational_state_circuit(
            initial_state=initial_state, qubits=model.qubits
        )
    elif name == "uniform_superposition":
        return uniform_superposition_state_circuit(
            initial_state=initial_state, qubits=model.qubits
        )
    elif name == "dicke":
        return dicke_state_circuit(n_electrons=model.n_electrons, qubits=model.qubits)
    elif name == "slater":
        return slater_state_circuit()
    elif name == "gaussian":
        return gaussian_state_circuit()
    else:
        raise NameError("No initial state named {}".format(name))
