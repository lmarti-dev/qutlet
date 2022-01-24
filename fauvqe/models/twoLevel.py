from __future__ import annotations

import importlib
from numbers import Real
import itertools

import numpy as np

from fauvqe.models.ising import Ising


class TwoLevel(Ising):
    """
    Class of independent two-level systems inherits Ising
    """
    
    def __init__(self, qubittype, n, h, t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        h  strength external field
        fields: basis of Single Qubit Pauli
        """
        super().__init__(
                    qubittype, 
                    np.array(n),
                    np.zeros((n[0], n[1])),
                    np.zeros((n[0], n[1])),
                    h,
                    "Z",
                    t
                )
    
    def copy(self) -> TwoLevel:
        self_copy = TwoLevel( self.qubittype,
                self.n,
                self.h[0],
                self.t )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy.hamiltonian = self.hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()
        
        return self_copy