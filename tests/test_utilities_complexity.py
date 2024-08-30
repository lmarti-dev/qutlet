import numpy as np
import pytest


from qutlet.utilities import (
    global_entanglement_wallach,
    bell_magic,
    chained_matrix_multiplication,
    normalize_vec,
)


def test_wallach_complexity():
    for n_qubits in (2, 3, 5):
        state_prod = np.zeros(2**n_qubits)
        state_prod[-1] = 1

        p_state_wall = global_entanglement_wallach(state_prod, n_qubits)
        print(f"product state: {p_state_wall}")

        assert np.isclose(p_state_wall, 0)

        state_entangled = np.zeros(2**n_qubits)

        state_entangled[0] = 1 / np.sqrt(2)
        state_entangled[-1] = 1 / np.sqrt(2)

        e_state_wall = global_entanglement_wallach(state_entangled, n_qubits)

        print(f"entangled state: {e_state_wall}")

        assert np.isclose(e_state_wall, 1)


def test_bell_magic():

    # all zeros
    for which in ("zeros", "y", "prod"):
        n_qubits = 2
        null_magic_state = np.zeros((2**n_qubits))

        if which == "zeros":
            pass
        elif which == "y":
            null_magic_state[-1] = 1j**n_qubits
        elif which == "prod":
            null_magic_state = np.ones_like(null_magic_state)
            null_magic_state = normalize_vec(null_magic_state)

        bm_prod = bell_magic(null_magic_state, 1000)
        bm_target = 0
        assert bm_prod == 0

    # case max
    theta = np.arccos(1 / np.sqrt(3))
    max_state_prod = chained_matrix_multiplication(
        np.kron,
        *[
            np.array([np.cos(theta / 2), np.exp(-1j * np.pi / 4) * np.sin(theta / 2)])
            for _ in range(n_qubits)
        ],
    )

    bm_prod = bell_magic(max_state_prod, 1000)
    bm_target = n_qubits * np.log2(27.0 / 11.0)
    assert bm_prod == bm_target
