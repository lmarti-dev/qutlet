from __future__ import annotations

import abc
from typing import Callable, Optional, Tuple


from fauvqe.models.abstractmodel import AbstractModel
import fauvqe.utils as utils
import openfermion as of 
import cirq

class FockModel(AbstractModel):
    r"""
    Implements a class that can run VQEs on occupation basis hamiltonians
    
    the Hamiltonian looks like 

    \( H = \sum_{ij} t_{ij} a^{\dagger}_i a_j + \sum_{ijkl} t_{ijkl} a^{\dagger}_{i} a^{dagger}_{j} a_k a_l \)

    openfermion has been admitted in this class, since it's not strictly a fermionic library, but
    also implements methods for bosons (and therefore has general fock utilities)


    There are a few conventions for naming variables here, and I should probably make a note of it
    to stay consistent
    model: references a FockModel child, so a full class instance with all the pots and pans
    fock_hamiltonian: a SymbolicOperator representing an actual physical hamiltonian
         but not a fermion hamiltonian because not an instance of FermionOperator
    fermion_hamiltonian: an instance of FermionOperator represneting a fermionic hamiltonian
    operator: Symbolic operator instance, not necessarily a complete hamiltonian, could simply be 1^0 say

    """
    def __init__(self,
                qubittype,
                n,
                encoding_name: str,
                qubit_maps: Tuple[Callable],
                fock_maps: Tuple[Callable]
                ):
        super().__init__(qubittype=qubittype,
                         n=n)
        self.fock_hamiltonian: of.SymbolicOperator=None
        self.encoding_name=encoding_name
        self.qubit_maps=qubit_maps
        self.fock_maps=fock_maps

        # get the fock hamiltonian
        self._set_fock_hamiltonian()
        # apply all function from the fock_maps to the hailtonian in order
        # take a look at the maps from fermiHubbard.py for examples
        # the underlying fock hamiltonian is flattened (even if defined on a grid)
        # so any fockmap function needs to take one variable only
        # you can also input an array, with the desired new order for indices
        # e.g. [2,3,1] will put the second item in the first place and so on
        # the simplest example is to move from udududud to uuuudddd (u=spin up d=spin down)
        self._apply_maps_to_fock_hamiltonian()
        # transform the fock hamiltonian (SymbolicOp) into a PauliSum
        # also use the qubit maps to do so.
        # qubit maps are *useless* don't use them
        self._set_hamiltonian()
        self.set_simulator("cirq")

    @property
    def flattened_qubits(self):
        """This function flattens the self.qubits (in case the qubittype is grid), since
        cirq has a lot of issues dealing with GridQubits in nested lists.
        Note that it simply returns a flattened list.
        ie

        1 2 3
        4 5 6       to         1 2 3 4 5 6 7 8 9
        7 8 9
        """
        if self.qubittype=="GridQubit":
            return utils.flatten_qubits(self.qubits)
        else:
            return self.qubits


    @abc.abstractmethod
    def _encode_fock_hamiltonian(self):
        raise NotImplementedError()# pragma: no cover

    @abc.abstractmethod
    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        raise NotImplementedError()# pragma: no cover

    @abc.abstractmethod
    def _get_initial_state(self):
        raise NotImplementedError()# pragma: no cover
    
    @abc.abstractmethod
    def _apply_maps_to_fock_hamiltonian(self,fock_maps):
        raise NotImplementedError()# pragma: no cover
    
    @property
    def flattened_mapped_qubits(self):
        return utils.flatten_qubits(self.apply_qubit_maps())
    
    def apply_qubit_maps(self):
        if self.qubit_maps is not None:
            qubits=self.qubits.copy()
            for qubit_map in self.qubit_maps:
                qubits = qubit_map(qubits)
            return qubits
        return self.qubits
    

    def energy(self):
        # this function should probably not do this, since it gives the actual energy of the hamiltonian
        # and not the energy of the calculated energy??
        hamiltonian_sparse = of.get_sparse_operator(self.fock_hamiltonian)
        # can be FermionOperator, QubitOperator, DiagonalCoulombHamiltonian, PolynomialTensor, BosonOperator, QuadOperator
        # this is the general openfermion method but it works for Boson and Fermions so I guess it can stay in fock
        return of.linalg.eigenspectrum(hamiltonian_sparse)

    def get_expectation(self,observables):
        return self.simulator.simulate_expectation_values(program=self.circuit,
                                                    observables=observables,
                                                    qubit_order=cirq.ops.QubitOrder.DEFAULT
                                                    )
    def add_missing_qubits(model):
        circuit_qubits=list(model.circuit.all_qubits())
        model_qubits = model.flattened_qubits
        missing_qubits = [x for x in model_qubits if x not in circuit_qubits]
        if circuit_qubits==[]:
            print("The circuit has no qubits")
            model.circuit = cirq.Circuit()
        model.circuit.append([cirq.I(mq) for mq in missing_qubits])

    def evaluate(self,observables: cirq.PauliSum):
        # this fails if the circuit has unused qubits, as they will not appear in circuit.qubits
        # and the validation methods will think the circuit has not the same qbits as the (hamiltonian) cirq.PauliSum
        
        try:
            expectation=self.get_expectation(observables)
        except ValueError:
            self.add_missing_qubits()
            expectation=self.get_expectation(observables)
        return expectation


