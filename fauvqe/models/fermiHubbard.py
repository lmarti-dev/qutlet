from __future__ import annotations


# External import

import openfermion as of
from typing import Dict,Union,Sequence,Optional
from cirq import OP_TREE,Qid
import numpy as np

# Internal import

from fauvqe.models.fermionicModel import FermionicModel
from tests.test_ising import hamiltonian


class FermiHubbardModel(FermionicModel):
    """
    implements VQEs on the Fermi-Hubbard Hamiltonian for a square lattice, using openfermion

    """
    def __init__(self,
                x_dimension: int,
                y_dimension: int,
                tunneling: float,
                coulomb: float,
                hamiltonian_options: Dict={}):
        
        self.x_dimension=x_dimension
        self.y_dimension=y_dimension
        self.tunneling=tunneling
        self.coulomb=coulomb
        self.hamiltonian_options=hamiltonian_options

        super().__init__(n=(x_dimension,y_dimension),
                        qubittype="GridQubit")
    def copy(self):
        self_copy = FermiHubbardModel(x_dimension=self.x_dimension,
                                    y_dimension=self.y_dimension,
                                    tunneling=self.tunneling,
                                    coulomb=self.coulomb,
                                    **self.hamiltonian_options)
        return self_copy
    def from_json_dict(self):
        pass
    def to_json_dict(self) -> Dict:
        pass
    def _set_fock_hamiltonian(self, reset: bool = True):
        """
        the fermi_hubbard function from openfermion represents the hamiltonian in a 1D array,
        so the ordering is already decided; and it is not snake, but end-to-end rows. 
        Perhaps I'll have to make my own since it's symbolic anyway (and not too hard to implement)
        The current operator ordering is from

        u11d11  u12d12  u13d13
        u21d21  u22d22  u23d23
        u31d31  u32d32  u33d33

        to

        0   1   2   3   4   5     6   7   8   9   10  11    12  13  14  15  16  17
        u11 d11 u12 d12 u13 d13 / u21 d21 u22 d22 u23 d23 / u31 d31 u32 d32 u33 d33

        """
        if reset:
            self.fock_hamiltonian=of.FermionOperator()
        self.fock_hamiltonian = of.fermi_hubbard(x_dimension=self.x_dimension,
                                                y_dimension=self.y_dimension,
                                                tunneling=self.tunneling,
                                                coulomb=self.coulomb,
                                                **self.hamiltonian_options)
        class_name=self.fock_hamiltonian.__class__.__name__
        assert (class_name == "FermionOperator"
        ), "fock_hamiltonian should be a FermionOperator, but is a {}".format(class_name)

    def get_initial_state(self,
                        name: str,
                        initial_state: Union[int, Sequence[int]] = 0
                        ) -> OP_TREE:
        # this is actually a fermionic gaussian state, need to find how to limit the occ number.
        if name == "slater":
            # with H = a*Ta + a*a*Vaa, get the T (one body) and V (two body) matrices from the hamiltonian
            tv=of.get_diagonal_coulomb_hamiltonian(self.fock_hamiltonian)
            # only get the T part
            quadratic_hamiltonian=of.QuadraticHamiltonian(tv.one_body)
            # get diagonalizing_bogoliubov_transform b_j = sum_i Q_ji a_i
            _,unitary_rows,_ = quadratic_hamiltonian.diagonalizing_bogoliubov_transform()
            # turns a matrix into a circuit of givens rotations
            #  you need the qubit system to get it however
            op_tree = of.prepare_slater_determinant(qubits=self.qubits,
                                                    slater_determinant_matrix=unitary_rows,
                                                    initial_state=initial_state)
            return op_tree          
        else:
            raise NameError("No initial state named {}".format(name))
