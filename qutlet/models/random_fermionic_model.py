from openfermion.ops import SymbolicOperator
from qutlet.models.fermionic_model import FermionicModel
from qutlet.utilities.fermion import (
    quadratic_hamiltonian_random_coefficients,
    quartic_hamiltonian_random_coefficients,
)


class RandomFermionicModel(FermionicModel):

    def __init__(
        self,
        n_qubits: int,
        neighbour_order: int = 2,
        term_order: int = 2,
        spin: bool = True,
        encoding_options: dict = None,
        **kwargs,
    ):
        if term_order not in [1, 2]:
            raise ValueError(
                f"Expected neighbour_order to be 1 or 2, got: {neighbour_order}"
            )
        self.neighbour_order = neighbour_order
        self.term_order = term_order
        self.spin = spin
        qubit_shape = (1, n_qubits)
        super().__init__(
            qubit_shape=qubit_shape, encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self) -> SymbolicOperator:
        self.fock_hamiltonian = quadratic_hamiltonian_random_coefficients(
            n_qubits=self.n_qubits, neighbour_order=self.neighbour_order, spin=self.spin
        )
        if self.term_order > 1:
            self.fock_hamiltonian += quartic_hamiltonian_random_coefficients(
                n_qubits=self.n_qubits,
                neighbour_order=self.neighbour_order,
                spin=self.spin,
            )

    @property
    def __to_json__(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "neighbour_order": self.neighbour_order,
            "term_order": self.term_order,
            "spin": self.spin,
            "encoding_options": self.encoding_options,
        }
