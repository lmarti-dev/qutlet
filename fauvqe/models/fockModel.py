from __future__ import annotations

import abc
from typing import Optional

# Internal import
from fauvqe.models.abstractmodel import AbstractModel
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
        

    @abc.abstractmethod
    def _encode_hamiltonian(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_initial_state(self):
        raise NotImplementedError()

  
    @abc.abstractmethod
    def set_circuit(self):
        raise NotImplementedError()

    
    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = self._encode_hamiltonian(self.fock_hamiltonian)

    
    def energy(self):
        hamiltonian_sparse = of.get_sparse_operator(self.fock_hamiltonian)
        # can be FermionOperator, QubitOperator, DiagonalCoulombHamiltonian, PolynomialTensor, BosonOperator, QuadOperator
        # this is the general openfermion method but it works for Boson and Fermions so I guess it can stay in fock
        return of.linalg.eigenspectrum(hamiltonian_sparse)