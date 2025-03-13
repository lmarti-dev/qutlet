from qutlet.models.qubit_model import QubitModel
from typing import Union, Iterable
import cirq
import itertools
import numpy as np
from qutlet.utilities.generic import default_value_handler


def ising_model(
    lattice_dimensions: tuple,
    h: float,
    J: float,
    qubits: list[cirq.Qid],
    interaction_pauli: cirq.Pauli,
    field_pauli: cirq.Pauli,
    periodic: bool,
    diagonal: bool,
):
    hamiltonian = cirq.PauliSum()
    ranges = (range(x) for x in lattice_dimensions)
    for site_a in itertools.product(*ranges):
        if diagonal:
            neighbours = itertools.product((0, 1), repeat=len(lattice_dimensions))
            # skip (0,0,0,0,0)
            next(neighbours)
        else:
            neighbours = itertools.permutations(
                (*(0,) * (len(lattice_dimensions) - 1), 1)
            )
        for site_b in neighbours:
            index_a = np.ravel_multi_index(multi_index=site_a, dims=lattice_dimensions)
            if periodic:
                site_neighbour = [
                    (a + b) % dim
                    for a, b, dim in zip(site_a, site_b, lattice_dimensions)
                ]
            else:
                site_neighbour = [a + b for a, b in zip(site_a, site_b)]
            if all(coor < dim for coor, dim in zip(site_neighbour, lattice_dimensions)):
                index_b = np.ravel_multi_index(
                    multi_index=site_neighbour, dims=lattice_dimensions
                )
                if not isinstance(h, (float, int)):
                    h = float(default_value_handler(shape=(1,), value=h))
                hamiltonian += (
                    -h
                    * interaction_pauli(qubits[index_a])
                    * interaction_pauli(qubits[index_b])
                )

    hamiltonian += -J * sum([field_pauli(q) for q in qubits])
    return hamiltonian


class IsingModel(QubitModel):
    def __init__(
        self,
        qubit_shape: Union[Iterable, int],
        h: float,
        J: float,
        interaction_pauli: cirq.Pauli = cirq.X,
        field_pauli: cirq.Pauli = cirq.Z,
        periodic: bool = False,
        diagonal: bool = False,
    ):
        self.h = h
        self.J = J
        self.interaction_pauli = interaction_pauli
        self.field_pauli = field_pauli
        self.periodic = periodic
        self.diagonal = diagonal

        self.eig_energies = None
        self.eig_states = None

        super().__init__(qubit_shape)

        self._set_hamiltonian()

    def _set_hamiltonian(self):
        self.hamiltonian = ising_model(
            lattice_dimensions=self.qubit_shape,
            h=self.h,
            J=self.J,
            qubits=self.qubits,
            interaction_pauli=self.interaction_pauli,
            field_pauli=self.field_pauli,
            periodic=self.periodic,
            diagonal=self.diagonal,
        )

    def __to_json__(self):
        return {
            "h": self.h,
            "J": self.J,
            "interaction_pauli": self.interaction_pauli,
            "field_pauli": self.field_pauli,
            "periodic": self.periodic,
            "diagonal": self.diagonal,
        }

    @property
    def ratio(self) -> float:
        return self.J / self.h

    @property
    def spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        if self.eig_energies is None or self.eig_states is None:
            self.eig_energies, self.eig_states = np.linalg.eigh(self.hamiltonian_matrix)
        return self.eig_energies, self.eig_states
