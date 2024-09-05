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
        is_spin_conserved: bool = True,
        encoding_options: dict = None,
        init_coefficients: list = None,
        **kwargs,
    ):
        if init_coefficients is None:
            init_coefficients = ("random",) * term_order
        if term_order not in [1, 2]:
            raise ValueError(
                f"Expected neighbour_order to be 1 or 2, got: {neighbour_order}"
            )
        self.neighbour_order = neighbour_order
        self.term_order = term_order
        self.is_spin_conserved = is_spin_conserved
        self.init_coefficients = init_coefficients
        super().__init__(
            qubit_shape=(1, n_qubits), encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self) -> SymbolicOperator:
        self.fock_hamiltonian = quadratic_hamiltonian_random_coefficients(
            n_qubits=self.n_qubits,
            neighbour_order=self.neighbour_order,
            is_spin_conserved=self.is_spin_conserved,
            coefficient=self.init_coefficients[0],
        )
        if self.term_order > 1:
            self.fock_hamiltonian += quartic_hamiltonian_random_coefficients(
                n_qubits=self.n_qubits,
                neighbour_order=self.neighbour_order,
                is_spin_conserved=self.is_spin_conserved,
                coefficient=self.init_coefficients[1],
            )

    @property
    def __to_json__(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "neighbour_order": self.neighbour_order,
            "term_order": self.term_order,
            "spin": self.is_spin_conserved,
            "encoding_options": self.encoding_options,
        }
