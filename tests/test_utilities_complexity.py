import numpy as np
import pytest


from qutlet.utilities import (
    global_entanglement_wallach,
    bell_magic,
    chained_matrix_multiplication,
    normalize_vec,
    stabilizer_renyi_entropy,
)

import itertools
from cirq import LineQubit, PauliSum, DensePauliString


def build_max_magic_state(n_qubits):
    paulis = ["I", "X", "Y", "Z"]
    pauli_strings = itertools.product(paulis, repeat=n_qubits)
    qubits = LineQubit.range(n_qubits)
    psum = PauliSum()
    dim = 2**n_qubits
    for pauli_string in pauli_strings:
        if pauli_string != "I" * n_qubits:
            psum += DensePauliString(pauli_string).on(*qubits) / (dim * (dim + 1))
        else:
            psum += DensePauliString(pauli_string).on(*qubits) / (dim)
    rho = psum.matrix(qubits=qubits)
    rho /= np.trace(rho)
    return rho


@pytest.mark.parametrize(("which"), ["prod", "max"])
def test_wallach_complexity(which):
    for n_qubits in (2, 3, 5):
        if which == "prod":
            state_prod = np.zeros(2**n_qubits)
            state_prod[-1] = 1

            p_state_wall = global_entanglement_wallach(state_prod, n_qubits)
            print(f"product state: {p_state_wall}")

            assert np.isclose(p_state_wall, 0)
        elif which == "max":
            state_entangled = np.zeros(2**n_qubits)

            state_entangled[0] = 1 / np.sqrt(2)
            state_entangled[-1] = 1 / np.sqrt(2)

            e_state_wall = global_entanglement_wallach(state_entangled, n_qubits)

            print(f"entangled state: {e_state_wall}")

            assert np.isclose(e_state_wall, 1)


@pytest.mark.parametrize(("which"), ["prod", "max"])
def test_stabilizer_renyi_entropy(which):
    n_qubits = 4
    dim = 2**n_qubits
    if which == "prod":
        state_prod = np.zeros(2**n_qubits, dtype="complex")
        state_prod[-1] = 1

        prod_renyi_entropy = stabilizer_renyi_entropy(state_prod, n_qubits)
        print(f"product state: {prod_renyi_entropy}")

        assert np.isclose(prod_renyi_entropy, 0)

    elif which == "max":
        # this doesn't work because rho is nonpositive.
        # but this is the only state for which I have an analytical value
        # of the rényi entropy.
        # what else can I test against..
        # rho = build_max_magic_state(n_qubits)
        theta = np.arccos(1 / np.sqrt(3))
        rho = chained_matrix_multiplication(
            np.kron,
            *[
                np.array(
                    [np.cos(theta / 2), np.exp(-1j * np.pi / 4) * np.sin(theta / 2)]
                )
                for _ in range(n_qubits)
            ],
        )
        prod_renyi_entropy = stabilizer_renyi_entropy(
            rho, n_qubits, alpha=2, normalize=False, ignore_nonpositive_rho=True
        )
        upper_bound = np.log(dim + 1) - np.log(2)

        assert np.isclose(prod_renyi_entropy, upper_bound)


@pytest.mark.parametrize(("which"), ["max", "zeros", "y", "prod"])
def test_bell_magic(which):

    # all zeros
    n_qubits = 2

    if which == "max":
        theta = np.arccos(1 / np.sqrt(3))
        max_state_prod = chained_matrix_multiplication(
            np.kron,
            *[
                np.array(
                    [np.cos(theta / 2), np.exp(-1j * np.pi / 4) * np.sin(theta / 2)]
                )
                for _ in range(n_qubits)
            ],
        )
        bm_prod = bell_magic(max_state_prod, 1000)
        bm_target = n_qubits * np.log2(27.0 / 11.0)
        assert bm_prod == bm_target
    else:
        null_magic_state = np.zeros((2**n_qubits), dtype="complex")
        if which == "zeros":
            null_magic_state[0] = 1
        elif which == "y":
            null_magic_state[-1] = 1j**n_qubits
        elif which == "prod":
            null_magic_state = np.ones_like(null_magic_state)
            null_magic_state = normalize_vec(null_magic_state)

        bm_prod = bell_magic(null_magic_state, 1000)
        bm_target = 0
        assert bm_prod == 0
