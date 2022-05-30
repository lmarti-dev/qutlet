from __future__ import annotations

import abc
from typing import Optional

# Internal import
from fauvqe.models.abstractmodel import AbstractModel
import fauvqe.utils as utils
import openfermion as of 

class FockModel(AbstractModel):
    r"""
    Implements a class that can run VQEs on occupation basis hamiltonians
    
    the Hamiltonian looks like 

    \( H = \sum_{ij} t_{ij} a^{\dagger}_i a_j + \sum_{ijkl} t_{ijkl} a^{\dagger}_{i} a^{dagger}_{j} a_k a_l \)

    openfermion has been admitted in this class, since it's not strictly a fermionic library, but
    also implements methods for bosons (and therefore has general fock utilities)

    """
    def __init__(self,
                qubittype,
                n):
        super().__init__(qubittype=qubittype,
                         n=n)
        self.fock_hamiltonian: of.SymbolicOperator=None
        self._set_fock_hamiltonian()
    
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
    def _encode_hamiltonian(self):
        raise NotImplementedError()# pragma: no cover

    @abc.abstractmethod
    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        raise NotImplementedError()# pragma: no cover

    @abc.abstractmethod
    def _get_initial_state(self):
        raise NotImplementedError()# pragma: no cover

    @abc.abstractmethod
    def set_circuit(self):
        raise NotImplementedError()# pragma: no cover
    @abc.abstractmethod
    def map_qubits(self,flattened_qubits):
        raise NotImplementedError()# pragma: no cover

    # this function should probably not do this, since it gives the actual energy of the hamiltonian
    # and not the energy of the calculated energy??
    def energy(self):
        hamiltonian_sparse = of.get_sparse_operator(self.fock_hamiltonian)
        # can be FermionOperator, QubitOperator, DiagonalCoulombHamiltonian, PolynomialTensor, BosonOperator, QuadOperator
        # this is the general openfermion method but it works for Boson and Fermions so I guess it can stay in fock
        return of.linalg.eigenspectrum(hamiltonian_sparse)
