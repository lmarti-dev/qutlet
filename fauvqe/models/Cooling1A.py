from __future__ import annotations

import importlib
from typing import Tuple, Dict, List
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.spinModel_fc import SpinModelFC


class Cooling1A(SpinModelFC):
    """
    Class for cooling structure using n system qubits and 1 ancilla qubit inherits SpinModelFC
    """
    
    def __init__(self, 
                 m_sys,
                 m_anc,
                 int_gates: List[cirq.PauliSum],
                 j_int: np.array,
                 t: Real = 0):
        """
        m_sys model defining the system of interest
        m_anc model defining the ancilla system
        int_gates, a cirq.PauliSum defining the interaction gates between system and ancillas including interaction constants
        t: Simulation Time
        """
        assert m_sys.qubittype == m_anc.qubittype, "Incompatible Qubittypes, System: {}, Ancilla: {}".format(
            m_sys.qubittype, m_anc.qubittype
        )
        assert m_anc.n[0] == m_sys.n[0] and m_anc.n[1] == 1, "Could not instantiate single row ancilla cooling with ancilla model of size {}".format(
            m_anc.n
        )
        j_int = np.array(j_int)
        assert j_int.shape == (len(int_gates), m_anc.n[0]), "Wrong shape of j_int, received: {} expected: {}".format(
            j_int.shape, (len(int_gates), m_anc.n[0])
        )
        
        n = [m_sys.n[0], m_sys.n[1] + 1]
        
        self.nbr_2Q_sys = len(m_sys._two_q_gates)
        self.nbr_2Q_anc = len(m_anc._two_q_gates)
        self.nbr_2Q_int = len(int_gates)
        self.nbr_2Q = self.nbr_2Q_sys + self.nbr_2Q_anc + self.nbr_2Q_int
        
        self.nbr_1Q_sys = len(m_sys._one_q_gates)
        self.nbr_1Q_anc = len(m_anc._one_q_gates)
        self.nbr_1Q = self.nbr_1Q_sys + self.nbr_1Q_anc
        
        self.sys_fc = issubclass(m_sys.__class__, SpinModelFC)
        self.anc_fc = issubclass(m_anc.__class__, SpinModelFC)
        
        two_q_gates = [*m_sys._two_q_gates, *m_anc._two_q_gates, *int_gates]
        one_q_gates = [*m_sys._one_q_gates, *m_anc._one_q_gates]
        
        j, h = _combine_jh(m_sys, m_anc, j_int)
        
        self.m_sys = m_sys
        self.m_anc = m_anc
        
        super().__init__(
            self, 
            m_sys.qubittype,
            np.array(n),
            j,
            h,
            two_q_gates: List[cirq.PauliSum],
            one_q_gates: List[cirq.PauliSum],
            t: Real = 0
            )
    
    def _combine_jh(m_sys, m_anc, j_int):
        n = [m_sys.n[0], m_sys.n[1] + 1]
        
        js = np.zeros(shape=(self.nbr_2Q, *n, *n))
        for g in range(self.nbr_2Q):
            if(g<self.nbr_2Q_sys):
                #System js
                if(self.sys_fc):
                    for i in range(m_sys.n[0]):
                        for j in range(m_sys.n[1]):
                            for l in range(j+1, m_sys.n[1], 1):
                                js[g, i, j, i, l] = m_sys.j[g, i, j, i, l]
                            for k in range(i+1, m_sys.n[0], 1):
                                for l in range(m_sys.n[1]):
                                    js[g, i, j, k, l] = m_sys.j[g, i, j, k, l]
                else:
                    for i in range(m_sys.n[0]-1):
                        for j in range(m_sys.n[1]-1):
                            js[g, i, j, i+1, j] = m_sys.j_v[g, i, j]
                            js[g, i, j, i, j+1] = m_sys.j_h[g, i, j]
                            js[g, i+1, j, i, j] = m_sys.j_v[g, i, j]
                            js[g, i, j+1, i, j] = m_sys.j_h[g, i, j]
                    for i in range(m_sys.n[0] - 1):
                        j = m_sys.n[1] - 1
                        js[g, i, j, i+1, j] = m_sys.j_v[g, i, j]
                        js[g, i+1, j, i, j] = m_sys.j_v[g, i, j]
                    for j in range(m_sys.n[1] - 1):
                        i = m_sys.n[0] - 1
                        js[g, i, j, i, j+1] = m_sys.j_h[g, i, j]
                        js[g, i, j+1, i, j] = m_sys.j_h[g, i, j]
                    if m_sys.boundaries[1] == 0:
                        for i in range(m_sys.n[0]):
                            j = m_sys.n[1] - 1
                            js[g, i, j, i, 0] = m_sys.j_h[g, i, j]
                            js[g, i, 0, i, j] = m_sys.j_h[g, i, j]
                    if m_sys.boundaries[0] == 0:
                        for j in range(m_sys.n[1]):
                            i = m_sys.n[0] - 1
                            js[g, i, j, 0, j] = m_sys.j_v[g, i, j]
                            js[g, 0, j, i, j] = m_sys.j_v[g, i, j]
            
            elif(g<self.nbr_2Q_sys + self.nbr_2Q_anc):
                #Ancilla js
                if(self.anc_fc):
                    for i in range(m_anc.n[0]):
                        for j in range(m_anc.n[1]):
                            for l in range(j+1, m_anc.n[1], 1):
                                js[g, i, m_sys.n[1] + j, i, m_sys.n[1] + l] = m_anc.j[g, i, j, i, l]
                            for k in range(i+1, m_anc.n[0], 1):
                                for l in range(m_anc.n[1]):
                                    js[g, i, m_sys.n[1] + j, k, m_sys.n[0] + l] = m_anc.j[g, i, j, k, l]
                else:
                    for i in range(m_anc.n[0] - 1):
                        js[g, i, m_sys.n[1], i+1, m_sys.n[1]] = m_anc.j_v[g, i, 0]
                        js[g, i + 1, m_sys.n[1], i, m_sys.n[1]] = m_anc.j_v[g, i, 0]
                    if(m_sys.boundaries[0] == 0):
                        i = m_anc.n[0] - 1
                        js[g, i, m_sys.n[1], 0, m_sys.n[1]] = m_anc.j_v[g, i, 0]
                        js[g, 0, m_sys.n[1], i, m_sys.n[1]] = m_anc.j_v[g, i, 0]
            else:
                #Interaction js
                for i in range(m_sys.n[0]):
                    for j in range(m_sys.n[1]):
                        js[g, i, j, i, m_sys.n[1]] = j_int[g, i]
                        js[g, i, m_sys.n[1], i, j] = j_int[g, i]
        
        h = np.zeros(shape=(self.nbr_1Q, *n))
        for g in range(self.nbr_1Q):
            if(g<self.nbr_2Q_sys):
                #System hs
                for i in range(m_sys.n[0]):
                    for j in range(m_sys.n[1]):
                        h[g, i, j] = m_sys.h[g, i, j]
            else:
                #Ancilla hs
                for i in range(m_anc.n[0]):
                    h[g, i, m_sys.n[1]] = m_anc.h[g, i, 0]
        
        return js, h
    
    def energy_2q(self, j) -> np.ndarray:
        n_sites = self.n[0] * self.n[1]
        Z = np.array([(-1) ** (np.arange(2 ** n_sites) >> i) for i in range(n_sites - 1, -1, -1)])
        
        ZZ_filter = np.zeros(
            2 ** (n_sites), dtype=np.float64
        )
        
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                #k==i, l>j
                for l in range(j+1, self.n[1], 1):
                    ZZ_filter += j[i, j, i, l] * Z[i * self.n[1] + j] * Z[i * self.n[1] + l]
                #k>i
                for k in range(i+1, self.n[0], 1):
                    for l in range(self.n[1]):
                        if ( (i<k) or (i==k and j<=l) ):
                            ZZ_filter += j[i, j, k, l] * Z[i * self.n[1] + j] * Z[k * self.n[1] + l]

        return ZZ_filter
    
    def energy_1q(self, h) -> np.ndarray:
        n_sites = self.n[0] * self.n[1]
        Z = np.array([(-1) ** (np.arange(2 ** n_sites) >> i) for i in range(n_sites - 1, -1, -1)])
        
        return h.reshape(n_sites).dot(Z)
    
    def energy(self, j, h) -> np.ndarray:
        return self.energy_1q(h) + self.energy_2q(j)
    
    def copy(self) -> SpinModel:
        self_copy = SpinModel( self.qubittype,
                self.n,
                np.transpose(self.j, (4, 0, 1, 2, 3)),
                np.transpose(self.h, (2, 0, 1)),
                self._two_q_gates,
                self._one_q_gates,
                self.t )

        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy.hamiltonian = self.hamiltonian.copy()

        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j": np.transpose(self.j, (4, 0, 1, 2, 3)),
                "h": np.transpose(self.h, (2, 0, 1)),
                "two_q_gates": self._two_q_gates,
                "one_q_gates": self._one_q_gates,
                "t": self.t
            },
            "params": {
                "circuit": self.circuit,
                "circuit_param": self.circuit_param,
                "circuit_param_values": self.circuit_param_values,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])

        inst.circuit = dct["params"]["circuit"]
        inst.circuit_param = dct["params"]["circuit_param"]
        inst.circuit_param_values = dct["params"]["circuit_param_values"]

        return inst