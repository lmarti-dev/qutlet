from __future__ import annotations

import abc
from typing import Callable, Optional, Tuple, Union, Sequence


from qutlet.models.qubitModel import QubitModel
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
        qubits,
        qubit_maps: Tuple[Callable] = None,
        fock_maps: Tuple[Callable] = None,
        encoding_options: dict,
    ):
        super().__init__(qubit_shape=qubits)
        self.fock_hamiltonian: of.SymbolicOperator = None
        self.qubit_maps = qubit_maps
        self.fock_maps = fock_maps
        self.encoding_options = encoding_options

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

    @abc.abstractmethod
    def _encode_fock_hamiltonian(self):
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def _set_fock_hamiltonian(self) -> of.SymbolicOperator:
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def _get_initial_state_circuit(self):
        raise NotImplementedError()  # pragma: no cover

    @abc.abstractmethod
    def _apply_maps_to_fock_hamiltonian(self, fock_maps):
        raise NotImplementedError()  # pragma: no cover

    def apply_qubit_maps(self):
        if self.qubit_maps is not None:
            qubits = self.qubits.copy()
            for qubit_map in self.qubit_maps:
                qubits = qubit_map(qubits)
            return qubits
        return self.qubits