from __future__ import annotations
from ctypes import Union
import openfermion as of
import cirq
import numpy as np
from typing import Callable, Optional, Tuple, Union, Sequence
import abc

from fauvqe.models.fockModel import FockModel
import fauvqe.utils as utils
import fauvqe.utils_cirq as cqutils


class FermionicModel(FockModel):
    """
    Fock model subclass that implements fermionic operators,
    i.e creators and annihilators which follow anticommutation rules.
    this class also implements the encoding (and not the fockmodel class),
    because the encodings are dependant on fermionic properties.
    This class is a pure class and not intended to be instanciated

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # this is a reminder of how up and down fermions are spread on the grid
        # the default value is the one given by the fermi_hubbard function, ie
        # u d u d

    @staticmethod
    def encode_model(fermion_hamiltonian, qubits, encoding_options) -> cirq.PauliSum:
        if "encoding_name" not in encoding_options.keys():
            raise KeyError("encoding_name missing")
        if "Z_snake" in encoding_options.keys():
            return FermionicModel._mapped_encode_model(
                fermion_hamiltonian=fermion_hamiltonian,
                qubits=qubits,
                encoding_name=encoding_options["encoding_name"],
                Z_snake=encoding_options["Z_snake"],
            )
        else:
            return FermionicModel._non_mapped_encode_model(
                fermion_hamiltonian=fermion_hamiltonian,
                qubits=qubits,
                encoding_name=encoding_options["encoding_name"],
            )

    @staticmethod
    def _mapped_encode_model(
        fermion_hamiltonian, qubits, encoding_name: str, Z_snake: Tuple
    ) -> cirq.PauliSum:
        encodings_dict = dict()
        encodings_dict["jordan_wigner"] = FermionicModel.mapped_jordan_wigner_fermion_operator
        if encoding_name in encodings_dict.keys():
            return of.qubit_operator_to_pauli_sum(
                encodings_dict[encoding_name](fermion_hamiltonian, Z_snake), qubits=qubits
            )
        else:
            raise KeyError(
                "No transform named {}. Allowed transforms: {}".format(
                    encoding_name, encodings_dict.keys()
                )
            )

    @staticmethod
    def _non_mapped_encode_model(fermion_hamiltonian, qubits, encoding_name: str) -> cirq.PauliSum:
        """
        use an openfermion transform to encode the occupation basis hamiltonian
        into a qubit hamiltonian.
        """
        encodings_dict = dict()
        encodings_dict["jordan_wigner"] = of.jordan_wigner
        encodings_dict["bravyi_kitaev"] = of.bravyi_kitaev
        encodings_dict["bravyi_kitaev_fast"] = cqutils.bravyi_kitaev_fast_wrapper

        if encoding_name in encodings_dict.keys():
            # need to specify the flattened_qubits here otherwise some validation methods will fail when evaluating the expectation
            # of the hamiltonian
            # this function is where all the mapping of qubits will happen
            # the encoding only does a symbolic encoding, regardless of the qubit structure
            # when the qubit operator is converted to pauli sum it determines which fermions is associated with each qubit
            return of.qubit_operator_to_pauli_sum(
                encodings_dict[encoding_name](fermion_hamiltonian), qubits=qubits
            )
        else:
            raise KeyError(
                "No transform named {}. Allowed transforms: {}".format(
                    encoding_name, encodings_dict.keys()
                )
            )

    @staticmethod
    def remap_fermion_hamiltonian(
        fermion_hamiltonian: of.SymbolicOperator, fock_map: Callable, fock_map_kwargs: dict() = None
    ):
        """Use a function on the fock hamilotnian's indices

        Args:
            fock_hamiltonian (of.SymbolicOperator): _description_
            fock_map (Callable): _description_
            fock_map_kwargs (dict, optional): _description_. Defaults to None.

        Returns:
            FermionOperator: The modified fermion operator
        """
        action_strings = fermion_hamiltonian.action_strings
        actions = fermion_hamiltonian.actions
        new_fermion_operator = of.FermionOperator()
        for terms, coeff in fermion_hamiltonian.terms.items():
            new_term = " ".join(
                (
                    str(fock_map(term[0], **fock_map_kwargs)) + action_strings[actions[term[1]]]
                    if fock_map_kwargs is not None
                    else str(fock_map(term[0])) + action_strings[actions[term[1]]]
                    for term in terms
                )
            )
            new_fermion_operator += of.FermionOperator(new_term, coeff)
        return sum(new_fermion_operator)

    @staticmethod
    def reindex_fermion_hamiltonian(
        fermion_hamiltonian: of.SymbolicOperator, fock_map_arr: Union[list, np.ndarray]
    ):
        """This function remaps the fock hamiltonian, in the same way as remap_fermion_hamiltonian
        however by using an array i,e given 3^0 and [1 3 0 2] one gets 1^2

        Args:
            fock_hamiltonian (of.SymbolicOperator): _description_
            fock_map_arr (Union[list,np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        flat_fock_map_arr = tuple(utils.flatten(fock_map_arr))
        FermionicModel.assert_map_matches_operator(fermion_hamiltonian, flat_fock_map_arr)
        action_strings = fermion_hamiltonian.action_strings
        actions = fermion_hamiltonian.actions
        new_fermion_operator = of.FermionOperator()
        for terms, coeff in fermion_hamiltonian.terms.items():
            new_term = " ".join(
                (
                    str(flat_fock_map_arr.index(term[0])) + action_strings[actions[term[1]]]
                    for term in terms
                )
            )
            new_fermion_operator += of.FermionOperator(new_term, coeff)
        return sum(new_fermion_operator)

    def _apply_maps_to_fock_hamiltonian(self):
        if self.fock_maps is not None:
            # check if we apply functions or just a reindexing array
            if isinstance(self.fock_maps[0], Callable):
                for fock_map in self.fock_maps:
                    self.fock_hamiltonian = self.remap_fermion_hamiltonian(
                        fermion_hamiltonian=self.fock_hamiltonian, fock_map=fock_map
                    )
            elif isinstance(self.fock_maps[0], list) or isinstance(self.fock_maps[0], int):
                self.fock_hamiltonian = self.reindex_fermion_hamiltonian(
                    fermion_hamiltonian=self.fock_hamiltonian, fock_map_arr=self.fock_maps
                )
            else:
                raise ValueError(
                    "expected fock_maps to be either a tuple of functions or a tuple of indices but got: {}".format(
                        type(self.fock_maps)
                    )
                )

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = self._encode_fock_hamiltonian()

    def _encode_fock_hamiltonian(self) -> cirq.PauliSum:
        return FermionicModel.encode_model(
            fermion_hamiltonian=self.fock_hamiltonian,
            qubits=self.flattened_qubits,
            encoding_options=self.encoding_options,
        )

    @staticmethod
    def mapped_jordan_wigner_fermion_operator(operator, Z_snake):
        # this is essentially the openfermion jw,
        # but modified in such a way that you can decide
        # exactly how the Z-snake moves
        # ie, given your fermion operator, which is "flat"
        # you can give a map to have your Z string follow a certain path
        # for example, the default Z string goes along
        #       0 1 2 3 4
        #       5 6 7 8 9
        #   but we want something like
        #       0 3 4 7 8
        #       1 2 5 6 9
        # you can supply the Z string array (the seocnd matrix) reordering
        # the Z_snake array follows the indices
        # original code belongs to openfermion

        FermionicModel.assert_map_matches_operator(operator, Z_snake)
        transformed_operator = of.QubitOperator()
        for term in operator.terms:
            # Initialize identity matrix.
            transformed_term = of.QubitOperator((), operator.terms[term])
            # Loop through operators, transform and multiply.
            for ladder_operator in term:
                z_factors = tuple(
                    (int(index), "Z")
                    for index in np.nonzero(np.array(Z_snake) < Z_snake[ladder_operator[0]])[0]
                )
                pauli_x_component = of.QubitOperator(z_factors + ((ladder_operator[0], "X"),), 0.5)
                if ladder_operator[1]:
                    pauli_y_component = of.QubitOperator(
                        z_factors + ((ladder_operator[0], "Y"),), -0.5j
                    )
                else:
                    pauli_y_component = of.QubitOperator(
                        z_factors + ((ladder_operator[0], "Y"),), 0.5j
                    )
                transformed_term *= pauli_x_component + pauli_y_component
            transformed_operator += transformed_term
        return transformed_operator

    @staticmethod
    def assert_map_matches_operator(operator, map_arr):
        assert set(map_arr) >= set(FermionicModel.get_ops_action_indices(operator))

    @staticmethod
    def get_ops_action_indices(operator):
        return list(set(utils.flatten(operator.terms.keys())))

    def jw_fermion_number_expectation(self, state):
        _,_,n_total_op=self.fermion_spin_number_operator()
        n_qubits = of.count_qubits(self.fock_hamiltonian)
        Nf=np.real(of.expectation(of.get_sparse_operator(n_total_op,n_qubits),state))
        return np.round(np.abs(Nf)).astype(int), np.sign(np.real(Nf)).astype(int)

    def fermion_spin_number_operator(self):
        n_qubits = of.count_qubits(self.fock_hamiltonian)
        n_up_op = sum([of.FermionOperator("{x}^ {x}".format(x=x)) for x in range(0,n_qubits,2)])
        n_down_op = sum([of.FermionOperator("{x}^ {x}".format(x=x)) for x in range(1,n_qubits,2)])
        n_total_op=sum(n_up_op,n_down_op)
        return n_up_op,n_down_op,n_total_op

    def get_encoded_terms(self, anti_hermitian: bool) -> "list[cirq.PauliSum]":
        operators = self.fock_hamiltonian.get_operators()
        encoded_terms = []
        parity = 1
        if anti_hermitian:
            parity = -1
        for operator in operators:
            operator = operator + parity * of.hermitian_conjugated(operator)
            encoded_term = FermionicModel.encode_model(
                operator, self.flattened_qubits, self.encoding_options
            )
            if encoded_term not in encoded_terms:
                encoded_terms.append(encoded_term)
        return encoded_terms

    def set_initial_state_circuit(
        self, name: str, Nf, initial_state: Union[int, Sequence[int]] = None
    ):
        """Inserts the cirq.OP_TREE generated by _get_initial_state into the circuit

        Args:
            name (str): the name of the type of initial state desired
            Nf: the number of fermions in the system
            initial_state (Union[int, Sequence[int]], optional): the indices of qubits that start n the 1 state. Defaults to 0 (i.e. all flipped down).
            An int input will be converted to binary and interpreted as a computational basis vector
            e.g. 34 = 100010 means the first and fifth qubits are initialized at one.
            rows (int): the rows taken from the Q matrix (rows of Q), where Q is defined from b* = Qa*, with a* creation operators.
                                                                Q diagonalizes Nf rows of the non-interacting hamiltonian
        """
        if initial_state == None:
            # default initial state is all 0s
            initial_state = []
        op_tree = self._get_initial_state(name=name, initial_state=initial_state, Nf=Nf)
        self.circuit.append(op_tree)
