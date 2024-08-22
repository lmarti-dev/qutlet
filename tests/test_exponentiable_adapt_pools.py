from qutlet.circuits.adapt_gate_pools import (
    QubitExcitationSet,
    FreeCouplersSet,
    ExponentiableGatePool,
)
from qutlet.circuits.adapt_vqe import exp_from_pauli_sum
from qutlet.utilities import all_pauli_str_commute
from qutlet.models import FermiHubbardModel


def exponentiate(gate_pool: ExponentiableGatePool):
    print(f"Gate pool size: {len(gate_pool.operator_pool)}")
    for ind, op in enumerate(gate_pool.operator_pool):
        print(op, end="")
        print(all_pauli_str_commute(op))
        exp_from_pauli_sum(op, theta=2)
        print("\r")


def test_qubit_excitations_set():
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2), tunneling=1, coulomb=2, n_electrons=[1, 1]
    )

    gate_pool = QubitExcitationSet(neighbour_order=model.n_qubits, qubits=model.qubits)
    exponentiate(gate_pool)


def test_free_coupler_set():
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2), tunneling=1, coulomb=2, n_electrons=[1, 1]
    )

    gate_pool = FreeCouplersSet(model=model)
    exponentiate(gate_pool)
