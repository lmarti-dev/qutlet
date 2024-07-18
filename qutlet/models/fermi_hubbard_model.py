import openfermion as of
import cirq
import numpy as np

from qutlet.models import FermionicModel
from qutlet.utilities import default_value_handler
from openfermion import FermionOperator, hermitian_conjugated
import itertools


def fermi_hubbard(
    lattice_dimensions: tuple,
    tunneling: float,
    coulomb: float,
    periodic: bool,
    diagonal: bool,
):
    hamiltonian = FermionOperator()
    dims = (*lattice_dimensions, 2)
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
            for spin in (0, 1):
                index_a = np.ravel_multi_index(multi_index=(*site_a, spin), dims=dims)
                if periodic:
                    site_neighbour = [
                        (a + b) % dim
                        for a, b, dim in zip(site_a, site_b, lattice_dimensions)
                    ]
                else:
                    site_neighbour = [a + b for a, b in zip(site_a, site_b)]
                if all(
                    coor < dim for coor, dim in zip(site_neighbour, lattice_dimensions)
                ):
                    index_b = np.ravel_multi_index(
                        multi_index=(*site_neighbour, spin), dims=dims
                    )
                    if not isinstance(tunneling, (float, int)):
                        tunneling = float(
                            default_value_handler(shape=(1,), value=tunneling)
                        )
                    hamiltonian += FermionOperator(
                        term=f"{index_a}^ {index_b}", coefficient=-tunneling
                    )

        index_up = np.ravel_multi_index(multi_index=(*site_a, 0), dims=dims)
        index_down = np.ravel_multi_index(multi_index=(*site_a, 1), dims=dims)
        if not isinstance(coulomb, (float, int)):
            coulomb = float(default_value_handler(shape=(1,), value=coulomb))

        # the 1/2 factor is to match openfermion's implementation
        hamiltonian += FermionOperator(
            term=f"{index_up}^ {index_up} {index_down}^ {index_down}",
            coefficient=0.5 * coulomb,
        )
    hamiltonian += hermitian_conjugated(hamiltonian)
    return hamiltonian


class FermiHubbardModel(FermionicModel):
    # bare * causes code to fail if everything is not keyworded
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
        # always lay those horizontally -> the shorter dimension goes in y and the longest gets the boundary
        # x_dimension (int): The width of the grid. y_dimension (int): The height of the grid
        # while n is the size of the qubit grid.
        self.lattice_dimensions = tuple(sorted(lattice_dimensions))
        self.tunneling = tunneling
        self.coulomb = coulomb
        self.periodic = periodic
        self.diagonal = diagonal
        # always stack the spins horizontally
        # ie
        #
        # u0 d0 u1 d1
        # u2 d2 u3 d3
        # flip for qubits so that indices are row-first

        if encoding_options is None:
            encoding_options = {"encoding_name": "jordan_wigner"}
        if encoding_options["encoding_name"] in (
            "jordan_wigner",
            "bravyi_kitaev",
        ):
            # we need twice the amount of qubits, we add a "spin" dimension
            qubit_shape = (*self.lattice_dimensions, 2)
        elif encoding_options["encoding_name"] in (
            "general_fermionic_encoding",
            "derby_klassen",
        ):
            raise NotImplementedError

        super().__init__(
            qubit_shape=qubit_shape, encoding_options=encoding_options, **kwargs
        )

    def _set_fock_hamiltonian(self, reset: bool = True):
        """This function sets the fock hamiltonian from the fermihubbard function in open fermion

        the fermi_hubbard function from openfermion represents the hamiltonian in a 1D array,
        so the ordering is already decided; and it is not snake, but end-to-end rows.
        The default operator ordering is from

        u11d11  u12d12  u13d13
        u21d21  u22d22  u23d23
        u31d31  u32d32  u33d33

        to

        0   1   2   3   4   5     6   7   8   9   10  11    12  13  14  15  16  17
        u11 d11 u12 d12 u13 d13 / u21 d21 u22 d22 u23 d23 / u31 d31 u32 d32 u33 d33

        Args:
            reset (bool, optional): Whether to reset the Hamiltonian to an empty FermionOperator. Defaults to True.
        """

        if reset:
            self.fock_hamiltonian = of.FermionOperator()
        self.fock_hamiltonian = fermi_hubbard(
            lattice_dimensions=self.lattice_dimensions,
            tunneling=self.tunneling,
            coulomb=self.coulomb,
            periodic=self.periodic,
            diagonal=self.diagonal,
        )

        # if for some reason the hamiltonian has no terms, turn it into an identity
        if not self.fock_hamiltonian.terms:
            self.fock_hamiltonian = of.FermionOperator.identity()

    @property
    def non_interacting_model(self):
        return FermiHubbardModel(
            lattice_dimensions=self.lattice_dimensions,
            tunneling=self.tunneling,
            coulomb=0.0,
            encoding_options=self.encoding_options,
            n_electrons=self.n_electrons,
        )

    def __to_json__(self) -> dict:
        return {
            "constructor_params": {
                "dimensions": self.lattice_dimensions,
                "n_electrons": self.n_electrons,
                "tunneling": self.tunneling,
                "coulomb": self.coulomb,
                "periodic": self.periodic,
                "encoding_options": self.encoding_options,
            },
        }

    @classmethod
    def from_dict(cls, dct: dict):
        inst = cls(**dct["constructor_params"])
        return inst

    @property
    def coulomb_model(self):
        if self.tunneling == 0:
            return self
        return FermiHubbardModel(
            lattice_dimensions=self.lattice_dimensions,
            tunneling=0.0,
            coulomb=self.coulomb,
            periodic=self.periodic,
            diagonal=self.diagonal,
            encoding_options=self.encoding_options,
            n_electrons=self.n_electrons,
        )
