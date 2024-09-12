from qutlet.models.fermionic_model import FermionicModel
from openfermion import FermionOperator, hermitian_conjugated
import itertools

import numpy as np

from qutlet.utilities.generic import default_value_handler


def tv_model(
    lattice_dimensions: tuple,
    tunneling: float,
    coulomb: float,
    periodic: bool,
    diagonal: bool,
):
    hamiltonian = FermionOperator()
    dims = lattice_dimensions
    ranges = (range(x) for x in lattice_dimensions)
    if not isinstance(coulomb, (float, int)):
        coulomb = float(default_value_handler(shape=(1,), value=coulomb))
    if not isinstance(tunneling, (float, int)):
        tunneling = float(default_value_handler(shape=(1,), value=tunneling))
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
            index_a = np.ravel_multi_index(multi_index=site_a, dims=dims)
            if periodic:
                site_neighbour = [
                    (a + b) % dim
                    for a, b, dim in zip(site_a, site_b, lattice_dimensions)
                ]
            else:
                site_neighbour = [a + b for a, b in zip(site_a, site_b)]
            if all(coor < dim for coor, dim in zip(site_neighbour, lattice_dimensions)):
                index_b = np.ravel_multi_index(multi_index=site_neighbour, dims=dims)

                hamiltonian += FermionOperator(
                    term=f"{index_a}^ {index_b}", coefficient=-tunneling
                )
                hamiltonian += FermionOperator(
                    term=f"{index_a}^ {index_a} {index_b}^ {index_b}",
                    coefficient=0.5 * coulomb,
                )

        # the 1/2 factor is to match openfermion's implementation

    hamiltonian += hermitian_conjugated(hamiltonian)
    return hamiltonian


class tVModel(FermionicModel):
    def __init__(
        self,
        *,
        lattice_dimensions: tuple,
        tunneling: float,
        coulomb: float,
        periodic: bool = False,
        diagonal: bool = False,
        encoding_options: dict = None,
        **kwargs,
    ):
        self.lattice_dimensions = tuple(sorted(lattice_dimensions))
        self.tunneling = tunneling
        self.coulomb = coulomb
        self.periodic = periodic
        self.diagonal = diagonal

        if encoding_options is None:
            encoding_options = {"encoding_name": "jordan_wigner"}
        if encoding_options["encoding_name"] in (
            "jordan_wigner",
            "bravyi_kitaev",
        ):
            qubit_shape = self.lattice_dimensions
        elif encoding_options["encoding_name"] in (
            "general_fermionic_encoding",
            "derby_klassen",
        ):
            raise NotImplementedError

        super().__init__(
            qubit_shape=qubit_shape, encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self):
        self.fock_hamiltonian = tv_model(
            lattice_dimensions=self.lattice_dimensions,
            tunneling=self.tunneling,
            coulomb=self.coulomb,
            periodic=self.periodic,
            diagonal=self.diagonal,
        )

        # if for some reason the hamiltonian has no terms, turn it into an identity
        if not self.fock_hamiltonian.terms:
            self.fock_hamiltonian = FermionOperator.identity()

    @property
    def __to_json__(self) -> dict:
        return {
            "dimensions": self.lattice_dimensions,
            "n_electrons": self.n_electrons,
            "tunneling": self.tunneling,
            "coulomb": self.coulomb,
            "periodic": self.periodic,
            "encoding_options": self.encoding_options,
        }
