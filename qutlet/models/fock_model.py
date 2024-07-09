from __future__ import annotations

import abc
from typing import Callable, Optional, Tuple, Union, Sequence


from models.qubit_model import QubitModel
from qutlet.utilities.generic import flatten
import openfermion as of
import cirq
import scipy


class FockModel(QubitModel):
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

    def __init__(
        self,
        *,
        qubit_shape,
        encoding_options: dict,
    ):
        super().__init__(qubit_shape=qubit_shape)
        self.fock_hamiltonian: of.SymbolicOperator = None
        self.encoding_options = encoding_options

        # get the fock hamiltonian
        self._set_fock_hamiltonian()
        self._set_hamiltonian()

    @abc.abstractmethod
    def _encode_fock_hamiltonian(self):
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        raise NotImplementedError()  # pragma: no cover
