from __future__ import annotations
from ctypes import Union
import openfermion as of
import cirq
import numpy as np
from typing import Callable, Optional, Tuple, Union, Sequence
import abc

from models.fock_model import FockModel
from qutlet.utilities import (
    flatten,
    bravyi_kitaev_fast_wrapper,
    index_bits,
    sum_even,
    jw_eigenspectrum_at_particle_number,
    sum_odd,
)

from openfermion import get_sparse_operator


class FermionicModel(FockModel):
    """
    Fock model subclass that implements fermionic operators,
    i.e creators and annihilators which follow anticommutation rules.
    this class also implements the encoding (and not the fockmodel class),
    because the encodings are dependant on fermionic properties.
    This class is a pure class and not intended to be instanciated

    """

    def __init__(self, system_fermions: list, **kwargs):
        # Number of fermions present in the system, initialized with the set_initial_state method
        # can be int or tuple of two ints with spin up and down fermions
        self.system_fermions = system_fermions
        super().__init__(**kwargs)
        # this is a reminder of how up and down fermions are spread on the grid
        # the default value is the one given by the fermi_hubbard function, ie
        # u d u d

    @staticmethod
    def encode_fermion_operator(
        fermion_hamiltonian, qubits, encoding_options: dict
    ) -> cirq.PauliSum:
        """
        use an openfermion transform to encode the occupation basis hamiltonian
        into a qubit hamiltonian.
        """
        encodings_dict = dict()
        encodings_dict["jordan_wigner"] = of.jordan_wigner
        encodings_dict["bravyi_kitaev"] = of.bravyi_kitaev
        encodings_dict["bravyi_kitaev_fast"] = bravyi_kitaev_fast_wrapper

        encoding_name = encoding_options["encoding_name"]

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

    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = self._encode_fock_hamiltonian()

    def _encode_fock_hamiltonian(self) -> cirq.PauliSum:
        return FermionicModel.encode_fermion_operator(
            fermion_hamiltonian=self.fock_hamiltonian,
            qubits=self.qubits,
            encoding_options=self.encoding_options,
        )

    @staticmethod
    def get_ops_action_indices(operator):
        return list(set(flatten(operator.terms.keys())))

    def jw_fermion_number_expectation(self, state):
        _, _, n_total_op = self.hamiltonian_spin_and_number_operator()
        n_qubits = of.count_qubits(self.fock_hamiltonian)
        system_fermions = np.real(
            of.expectation(of.get_sparse_operator(n_total_op, n_qubits), state)
        )
        return np.round(np.abs(system_fermions)).astype(int), np.sign(
            np.real(system_fermions)
        ).astype(int)

    @staticmethod
    def spin_and_number_operator(n_qubits: int):
        n_up_op = of.FermionOperator()
        for x in range(0, n_qubits, 2):
            n_up_op += of.FermionOperator("{x}^ {x}".format(x=x))

        n_down_op = of.FermionOperator()
        for x in range(1, n_qubits, 2):
            n_down_op += of.FermionOperator("{x}^ {x}".format(x=x))

        n_total_op = sum(n_up_op, n_down_op)
        return n_up_op, n_down_op, n_total_op

    def hamiltonian_spin_and_number_operator(self):
        n_qubits = of.count_qubits(self.fock_hamiltonian)
        return self.spin_and_number_operator(n_qubits=n_qubits)

    def get_encoded_terms(self, anti_hermitian: bool) -> "list[cirq.PauliSum]":
        operators = self.fock_hamiltonian.get_operators()
        encoded_terms = []
        parity = 1
        if anti_hermitian:
            parity = -1
        for operator in operators:
            operator = operator + parity * of.hermitian_conjugated(operator)
            encoded_term = FermionicModel.encode_fermion_operator(
                operator, self.qubits, self.encoding_options
            )
            if encoded_term not in encoded_terms:
                encoded_terms.append(encoded_term)
        return encoded_terms

    def get_quadratic_hamiltonian_wrapper(self, fermion_hamiltonian):
        # not sure this is correct
        # but in case the fermion operator is null (ie empty hamiltonian, get a zeros matrix)
        if fermion_hamiltonian == of.FermionOperator.identity():
            return of.QuadraticHamiltonian(np.zeros((np.prod(self.n), np.prod(self.n))))
        quadratic_hamiltonian = of.get_quadratic_hamiltonian(
            fermion_hamiltonian, ignore_incompatible_terms=True
        )
        return quadratic_hamiltonian

    @property
    def spectrum(self):
        eig_energies, eig_states = jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(self.fock_hamiltonian),
            particle_number=self.n_electrons,
            expanded=True,
        )
        return eig_energies, eig_states
