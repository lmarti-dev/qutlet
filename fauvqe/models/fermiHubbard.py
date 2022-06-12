from __future__ import annotations

# External import

import openfermion as of
from typing import Callable, Dict, Tuple,Union,Sequence,Optional
from typing import get_origin
import cirq
import numpy as np


# Internal import

from fauvqe.models.fermionicModel import FermionicModel
import fauvqe.utils as utils

class FermiHubbardModel(FermionicModel):
    """
    implements VQEs on the Fermi-Hubbard Hamiltonian for a square lattice, using openfermion

    """
    def __init__(self,
                x_dimension: int,
                y_dimension: int,
                tunneling: float,
                coulomb: float,
                hamiltonian_options: Dict={},
                hv_grid_qubits=0,
                encoding_name: str="jordan_wigner",
                qubit_maps: Tuple[Callable]= None,
                fock_maps: Tuple= None,
                Z_snake: Tuple=None
                ):
        
        self.x_dimension=x_dimension
        self.y_dimension=y_dimension
        self.tunneling=tunneling
        self.coulomb=coulomb
        self.hv_grid_qubits = hv_grid_qubits
        self.hamiltonian_options=hamiltonian_options


        # decide whether to stack the spins horizontally or vertically
        # ie
        #
        # u d u d
        # u d u d
        # 
        # or
        # 
        # u d
        # d u
        # u d
        # d u
        #

        if self.hv_grid_qubits==0:
            # horizontally by default
            n=(x_dimension,2*y_dimension)
        elif self.hv_grid_qubits==1:
            (2*x_dimension,y_dimension)
        else:
            raise ValueError("Invalid value for hv_grid_qubit. expected 0 or 1 but got {}".format(hv_grid_qubits))
        
        super().__init__(n=n,
                        qubittype="GridQubit",
                        encoding_name=encoding_name,
                        qubit_maps=qubit_maps,
                        fock_maps=fock_maps,
                        Z_snake=Z_snake)
        



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
            self.fock_hamiltonian=of.FermionOperator()
        self.fock_hamiltonian = of.fermi_hubbard(x_dimension=self.x_dimension,
                                                y_dimension=self.y_dimension,
                                                tunneling=self.tunneling,
                                                coulomb=self.coulomb,
                                                **self.hamiltonian_options)
        class_name=self.fock_hamiltonian.__class__.__name__
        assert (class_name == "FermionOperator"
        ), "fock_hamiltonian should be a FermionOperator, but is a {}".format(class_name)

    def _get_initial_state(self,
                        name: str,
                        initial_state: Union[int, Sequence[int]],
                        Nf: int
                        ) -> cirq.OP_TREE:
        if name == "gaussian":
            tv=of.get_diagonal_coulomb_hamiltonian(self.fock_hamiltonian)
            quadratic_hamiltonian=of.QuadraticHamiltonian(tv.one_body)
            op_tree = of.prepare_gaussian_state(qubits=self.flattened_qubits,
                                    quadratic_hamiltonian=quadratic_hamiltonian,
                                    occupied_orbitals=list(range(Nf)),
                                    initial_state=initial_state)
            return op_tree
        elif name == "slater":
            _,unitary_rows = self.diagonalize_non_interacting_hamiltonian()

            op_tree=of.prepare_slater_determinant(qubits=self.flattened_qubits,
                                        slater_determinant_matrix=unitary_rows[:Nf,:],
                                        initial_state=initial_state)
            return op_tree
        else:   
            raise NameError("No initial state named {}".format(name))
    


    def set_initial_state_circuit(self,name: str,
                            initial_state: Union[int, Sequence[int]] = None,
                            Nf: int=None):
        """Inserts the cirq.OP_TREE generated by _get_initial_state into the circuit

        Args:
            name (str): the name of the type of initial state desired
            initial_state (Union[int, Sequence[int]], optional): the indices of qubits that start n the 1 state. Defaults to 0 (i.e. all flipped down).
                                                                An int input will be converted to binary and interpreted as a computational basis vector
                                                                e.g. 34 = 100010 means the first and fifth qubits are initialized at one.
            rows (int): the rows taken from the Q matrix (rows of Q), where Q is defined from b* = Qa*, with a* creation operators. 
                                                                Q diagonalizes Nf rows of the non-interacting hamiltonian
        """
        if Nf==None:
            # if Nf is none, use half the Q matrix
            Nf = self.x_dimension*self.y_dimension
        if initial_state == None:
            # default initial state is all 0s
            initial_state = []
        cirq.OP_TREE=self._get_initial_state(name=name,initial_state=initial_state,Nf=Nf)
        self.circuit.append(cirq.OP_TREE)

    def diagonalize_non_interacting_hamiltonian(self):
        # with H = a*Ta + a*a*Vaa, get the T (one body) and V (two body) matrices from the hamiltonian
        tv=of.get_diagonal_coulomb_hamiltonian(self.fock_hamiltonian)
        """
            only get the T part
            the repartition of spin sectors on the matrix is as such:
                u11 d11 u12 d12 u21 d21 u22 d22
            u11 0   0   t   0   t   0   0   0
            d11 0   0   0   t   0   t   0   0
            u12
            d12      ...
            u21
            d21
            u22
            d22
        """
        quadratic_hamiltonian=of.QuadraticHamiltonian(tv.one_body)
        # get diagonalizing_bogoliubov_transform b_j = sum_i Q_ji a_i s.t H = bDb* with D diag.
        # the bogoliubov transform conserves particle number, i.e. the bogops are single particle (i think?)
        orbital_energies,unitary_rows,_ = quadratic_hamiltonian.diagonalizing_bogoliubov_transform()
        
        # sort them so that you get them in order 
        idx = np.argsort(orbital_energies)
        
        unitary_rows = unitary_rows[idx,:]
        orbital_energies = orbital_energies[idx]

        return orbital_energies,unitary_rows

    @property
    def non_interacting_fock_hamiltonian(self) -> of.FermionOperator:
        return of.fermi_hubbard(x_dimension=self.x_dimension,
                                y_dimension=self.y_dimension,
                                tunneling=self.tunneling,
                                coulomb=0.0,
                                **self.hamiltonian_options)
    @property
    def non_interacting_hamiltonian(self) -> cirq.PauliSum:
        return FermiHubbardModel.encode_model(fermion_hamiltonian=self.non_interacting_fock_hamiltonian, 
                                            qubits=self.flattened_qubits,
                                            encoding_name=self.encoding_name)
    @staticmethod
    def snake_map(n,dimx,dimy):
        (x,y) = utils.linear_to_grid(n=n,dimx=dimx,dimy=dimy)
        (x,y) = utils.arg_flip_cross_row(x=x,y=y,dimy=dimy)
        return utils.grid_to_linear(x=x,y=y,dimx=dimx,dimy=dimy)
    @staticmethod
    def altsec_map(n,dimx,dimy):
        (x,y) = utils.linear_to_grid(n=n,dimx=dimx,dimy=dimy)
        yp = utils.arg_alternating_index_to_sector(index=y,N=dimy)
        return utils.grid_to_linear(x=x,y=yp,dimx=dimx,dimy=dimy)
        
    def pretty_print_jw_order(self,pauli_string: cirq.PauliString): #pragma: no cover
        last_qubit = max(self.flattened_qubits)
        mat = np.array([["0" 
                        for y in range(last_qubit.col+1)] 
                        for x in range(last_qubit.row+1)])
        for k,v in pauli_string.items():
            mat[(k.row,k.col)]=v
        mat=mat.tolist()
        print("\n".join(["".join(row) for row in mat]))