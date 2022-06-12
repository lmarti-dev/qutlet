from __future__ import annotations
from ctypes import Union
from curses import has_key

# External import

import openfermion as of
import cirq
import numpy as np
from typing import Callable, Tuple

# Internal import
from fauvqe.models.fockModel import FockModel
import fauvqe.utils as utils


class FermionicModel(FockModel):
    """
    Fock model subclass that implements fermionic operators, 
    i.e creators and annihilators which follow anticommutation rules.
    this class also implements the encoding (and not the fockmodel class),
    because the encodings are dependant on fermionic properties

    """
    def __init__(self,
                qubittype: str,
                n,
                encoding_name: str,
                qubit_maps:Tuple[Callable]=None,
                fock_maps: Tuple=None,
                Z_snake: Tuple=None
                ):
        self.Z_snake=Z_snake
        super().__init__(qubittype=qubittype,
                        n=n,
                        encoding_name=encoding_name,
                        qubit_maps=qubit_maps,
                        fock_maps=fock_maps)
        
        # this is a reminder of how up and down fermions are spread on the grid
        # the default value is the one given by the fermi_hubbard function, ie
        # u d u d
    @staticmethod
    def mapped_encode_model(fermion_hamiltonian,qubits,encoding_name: str,Z_snake: Tuple) -> cirq.PauliSum:
        encodings_dict=dict()
        encodings_dict["jordan_wigner"]=FermionicModel.mapped_jordan_wigner_fermion_operator
        if encoding_name in encodings_dict.keys():
            return of.qubit_operator_to_pauli_sum(encodings_dict[encoding_name](fermion_hamiltonian,Z_snake),qubits=qubits)
        else:
            raise KeyError("No transform named {}. Allowed transforms: {}".format(encoding_name,encodings_dict.keys()))   
    
    @staticmethod
    def encode_model(fermion_hamiltonian,qubits,encoding_name: str) -> cirq.PauliSum:
        """
        use an openfermion transform to encode the occupation basis hamiltonian
        into a qubit hamiltonian.
        """
        encodings_dict=dict()
        encodings_dict["jordan_wigner"]=of.jordan_wigner
        encodings_dict["bravyi_kitaev"]=of.bravyi_kitaev
        
        if encoding_name in encodings_dict.keys():
            # need to specify the flattened_qubits here otherwise some validation methods will fail when evaluating the expectation
            # of the hamiltonian
            # this function is where all the mapping of qubits will happen
            # the encoding only does a symbolic encoding, regardless of the qubit structure
            # when the qubit operator is converted to pauli sum it determines which fermions is associated with each qubit
            return of.qubit_operator_to_pauli_sum(encodings_dict[encoding_name](fermion_hamiltonian),
                                                qubits=qubits)
        else:
            raise KeyError("No transform named {}. Allowed transforms: {}".format(encoding_name,encodings_dict.keys()))

    @staticmethod
    def remap_fermion_hamiltonian(fermion_hamiltonian: of.SymbolicOperator,fock_map: Callable ,fock_map_kwargs: dict()=None):
        """Use a function on the fock hamilotnian's indices

        Args:
            fock_hamiltonian (of.SymbolicOperator): _description_
            fock_map (Callable): _description_
            fock_map_kwargs (dict, optional): _description_. Defaults to None.

        Returns:
            FermionOperator: The modified fermion operator
        """
        action_strings=fermion_hamiltonian.action_strings
        actions = fermion_hamiltonian.actions
        new_fermion_operator=of.FermionOperator()
        for terms,coeff in fermion_hamiltonian.terms.items():
            new_term = " ".join((str(fock_map(term[0],**fock_map_kwargs)) + action_strings[actions[term[1]]]
                                if fock_map_kwargs is not None 
                                else str(fock_map(term[0])) + action_strings[actions[term[1]]]
                                for term in terms))
            new_fermion_operator+=(of.FermionOperator(new_term,coeff))
        return sum(new_fermion_operator)
    @staticmethod
    def reindex_fermion_hamiltonian(fermion_hamiltonian: of.SymbolicOperator,fock_map_arr: Union[list,np.ndarray]):
        """This function remaps the fock hamiltonian, in the same way as remap_fermion_hamiltonian
        however by using an array i,e given 3^0 and [1 3 0 2] one gets 1^2

        Args:
            fock_hamiltonian (of.SymbolicOperator): _description_
            fock_map_arr (Union[list,np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        flat_fock_map_arr = tuple(utils.flatten(fock_map_arr))
        FermionicModel.assert_map_matches_operator(flat_fock_map_arr,fermion_hamiltonian)
        action_strings=fermion_hamiltonian.action_strings
        actions = fermion_hamiltonian.actions
        new_fermion_operator=of.FermionOperator()
        for terms,coeff in fermion_hamiltonian.terms.items():
            new_term = " ".join(
                                (str(flat_fock_map_arr.index(term[0]))
                                 + action_strings[actions[term[1]]] 
                                 for term in terms))
            new_fermion_operator+=(of.FermionOperator(new_term,coeff))
        return sum(new_fermion_operator)
    
    def _apply_maps_to_fock_hamiltonian(self):
        if self.fock_maps is not None:
            # check if we apply functions or just a reindexing array
            if isinstance(self.fock_maps[0], Callable):
                for fock_map in self.fock_maps:
                    self.fock_hamiltonian = self.remap_fermion_hamiltonian(fermion_hamiltonian=self.fock_hamiltonian,fock_map=fock_map)
            elif isinstance(self.fock_maps[0],list) or isinstance(self.fock_maps[0],int):
                    self.fock_hamiltonian = self.reindex_fermion_hamiltonian(fermion_hamiltonian=self.fock_hamiltonian,fock_map_arr=self.fock_maps)
            else:
                raise ValueError(
                    "expected fock_maps to be either a tuple of functions or a tuple of indices but got: {}".format(type(self.fock_maps)))
    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = self._encode_fock_hamiltonian()

    def _encode_fock_hamiltonian(self) -> cirq.PauliSum:
        if self.Z_snake is not None:
            return self.mapped_encode_model(fermion_hamiltonian=self.fock_hamiltonian,
                                            qubits=self.flattened_mapped_qubits,
                                            encoding_name=self.encoding_name,
                                            Z_snake=self.Z_snake
                                            )
        return self.encode_model(fermion_hamiltonian=self.fock_hamiltonian,
                                            qubits=self.flattened_mapped_qubits,
                                            encoding_name=self.encoding_name
                                            )

    def set_circuit(self,
                    qalgorithm: str
                    ):
        # set fermions on qubits -> initial states -> h/v hopping - fswap loop -> onsite 
        if qalgorithm == "hva":
            raise NotImplementedError("Hamiltonian Variational Ansatz doesn't exist yet")
        else:
            raise NameError("{} is not a valid circuit".format(qalgorithm))
    
    @staticmethod
    def mapped_jordan_wigner_fermion_operator(operator,Z_snake):
        # this is essentially the openfermion jw,
        # but modified in such a way that you can decide
        # exactly how the Z-snake moves
        # ie, given your fermion operator, which is "flat"
        # you can give a map to have your Z string follow a certain path
        # for example, the defualt Z string goes along
        #       0 1 2 3 4
        #       5 6 7 8 9
        #   but we want something like
        #       0 3 4 7 8
        #       1 2 5 6 9
        # you can supply the Z string array reordeing
        # the Z_snake array follows the indices 
        # original code belongs to openfermion

        #flatten it in case it's given as [[1 2][4 3]]
        flat_Z_snake = tuple(utils.flatten(Z_snake))
        FermionicModel.assert_map_matches_operator(flat_Z_snake,operator)
        transformed_operator = of.QubitOperator()
        for term in operator.terms:
            # Initialize identity matrix.
            transformed_term = of.QubitOperator((), operator.terms[term])
            # Loop through operators, transform and multiply.
            for ladder_operator in term:
                z_factors = tuple(
                    (index, 'Z') for index in flat_Z_snake[:flat_Z_snake.index(ladder_operator[0])])
                pauli_x_component = of.QubitOperator(
                    z_factors + ((ladder_operator[0], 'X'),), 0.5)
                if ladder_operator[1]:
                    pauli_y_component = of.QubitOperator(
                        z_factors + ((ladder_operator[0], 'Y'),), -0.5j)
                else:
                    pauli_y_component = of.QubitOperator(
                        z_factors + ((ladder_operator[0], 'Y'),), 0.5j)
                transformed_term *= pauli_x_component + pauli_y_component
            transformed_operator += transformed_term
        return transformed_operator

    @staticmethod
    def assert_map_matches_operator(map_arr,operator):
        assert np.array(sorted(map_arr) 
                == sorted(FermionicModel.get_ops_action_indices(operator))).all()

    @staticmethod
    def get_ops_action_indices(operator):
        return list(set(utils.flatten(operator.terms.keys())))