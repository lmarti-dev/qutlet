import openfermion as of
from typing import Callable, Dict, Tuple, Union, Sequence, Optional
import cirq
import numpy as np
import copy

from qutlet.models.fermionicModel import FermionicModel

from qutlet.utilities.generic import index_bits


class FermiHubbardModel(FermionicModel):
    # bare * causes code to fail if everything is not keyworded
    def __init__(
        self,
        *,
        x_dimension: int,
        y_dimension: int,
        tunneling: float,
        coulomb: float,
        hamiltonian_options: Dict = {"periodic": False},
        encoding_options: Dict = None,
        **kwargs
    ):
        # always lay those horizontally -> the shorter dimension goes in y and the longest gets the boundary
        # x_dimension (int): The width of the grid. y_dimension (int): The height of the grid
        # while n is the size of the qubit grid.
        self.x_dimension = min((x_dimension, y_dimension))
        self.y_dimension = max((x_dimension, y_dimension))
        self.tunneling = tunneling
        self.coulomb = coulomb
        self.hamiltonian_options = hamiltonian_options
        # always stack the spins horizontally
        # ie
        #
        # u0 d0 u1 d1
        # u2 d2 u3 d3
        # flip for qubits so that indices are row-first

        if encoding_options is None:
            # z-snake is only relevant for 1d encoding like jordan-wigner or bravyi kitaev
            encoding_options = {"encoding_name": "jordan_wigner"}
            # encoding_options["Z_snake"]=self.common_Z_snakes(name="weaved_double_s",dimx=self.x_dimension,dimy=self.y_dimension)
            # moves the spins into sectors ududud -> uuuddd (only along the horizontal axis)
        # if "fock_maps" not in kwargs.keys():
        #     kwargs["fock_maps"] = alternating_indices_to_sectors(np.reshape(np.arange(np.prod(n)),n),axis=1).tolist()
        #     # this "default" jw setup moves the spins on one side of the qubit grid, and create a Z_snake that makes hopping computation easy
        if encoding_options["encoding_name"] in (
            "jordan_wigner",
            "bravyi_kitaev",
        ):
            # derby-klassen uses 1.5N qubits in a checkerboard, but here we'll go with 2N and remove a quarter, that's easier
            # considering how abstractmodel is setup,
            qubits = (self.y_dimension, 2 * self.x_dimension)
        elif encoding_options["encoding_name"] in (
            "general_fermionic_encoding",
            "derby_klassen",
        ):
            raise NotImplementedError

        super().__init__(qubits=qubits, encoding_options=encoding_options, **kwargs)

    def copy(self):
        self_copy = copy.deepcopy(self)
        return self_copy

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
        self.fock_hamiltonian = of.fermi_hubbard(
            x_dimension=self.x_dimension,
            y_dimension=self.y_dimension,
            tunneling=self.tunneling,
            coulomb=self.coulomb,
            **self.hamiltonian_options
        )

        # if for some reason the hamiltonian has no terms, turn it into an identity
        if not self.fock_hamiltonian.terms:
            self.fock_hamiltonian = of.FermionOperator.identity()

    @property
    def non_interacting_model(self):
        return FermiHubbardModel(
            x_dimension=self.x_dimension,
            y_dimension=self.y_dimension,
            tunneling=self.tunneling,
            coulomb=0.0,
            encoding_options=self.encoding_options,
            qubit_maps=self.qubit_maps,
            fock_maps=self.fock_maps,
            hamiltonian_options=self.hamiltonian_options,
        )

    def pretty_print_jw_order(self, pauli_string: cirq.PauliString):  # pragma: no cover
        last_qubit = max(self.qubits)
        mat = np.array(
            [
                ["0" for y in range(last_qubit.col + 1)]
                for x in range(last_qubit.row + 1)
            ]
        )

        for k, v in pauli_string.items():
            mat[(k.row, k.col)] = v
        mat = mat.tolist()
        print(pauli_string)
        print("\n".join(["".join(row) for row in mat]))

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "x_dimension": self.x_dimension,
                "y_dimension": self.y_dimension,
                "tunneling": self.tunneling,
                "coulomb": self.coulomb,
                "hamiltonian_options": self.hamiltonian_options,
                "encoding_options": self.encoding_options,
            },
            "params": {
                "circuit": self.circuit,
                "circuit_param": self.circuit_param,
                "circuit_param_values": self.circuit_param_values,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])

        inst.circuit = dct["params"]["circuit"]
        inst.circuit_param = dct["params"]["circuit_param"]
        inst.circuit_param_values = dct["params"]["circuit_param_values"]

        return inst
