from __future__ import annotations

import importlib
from typing import Tuple, Dict, List
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.spinModel_fc import SpinModelFC


class CoolingNA(SpinModelFC):
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
        assert m_anc.n[0] == m_sys.n[0] and m_anc.n[1] == m_sys.n[1], "Could not instantiate system doubled ancilla cooling with ancilla model of size {}".format(
            m_anc.n
        )
        j_int = np.array(j_int)
        assert j_int.shape == (len(int_gates), *m_anc.n ), "Wrong shape of j_int, received: {} expected: {}".format(
            j_int.shape, (len(int_gates), m_anc.n[0])
        )
        
        n = [2*m_sys.n[0], m_sys.n[1]]
        
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
        
        j, h = self._combine_jh(m_sys, m_anc, j_int)
        
        self.m_sys = m_sys
        self.m_anc = m_anc
        self.j_int = j_int
        
        super().__init__( 
            m_sys.qubittype,
            np.array(n),
            j,
            h,
            two_q_gates,
            one_q_gates,
            t
            )
    
    def _combine_jh(self, m_sys, m_anc, j_int):
        n = [2*m_sys.n[0], m_sys.n[1]]
        
        js = np.zeros(shape=(self.nbr_2Q, *n, *n))
        for g in range(self.nbr_2Q):
            if(g<self.nbr_2Q_sys):
                #System js
                if(self.sys_fc):
                    for i in range(m_sys.n[0]):
                        for j in range(m_sys.n[1]):
                            for l in range(j+1, m_sys.n[1], 1):
                                js[g, i, j, i, l] = m_sys.j[i, j, i, l, g]
                                js[g, i, l, i, j] = m_sys.j[i, j, i, l, g]
                            for k in range(i+1, m_sys.n[0], 1):
                                for l in range(m_sys.n[1]):
                                    js[g, i, j, k, l] = m_sys.j[i, j, k, l, g]
                                    js[g, k, l, i, j] = m_sys.j[i, j, k, l, g]
                else:
                    for i in range(m_sys.n[0]-1):
                        for j in range(m_sys.n[1]-1):
                            js[g, i, j, i+1, j] = m_sys.j_v[i, j, g]
                            js[g, i, j, i, (j+1)] = m_sys.j_h[i, j, g]
                            js[g, i+1, j, i, j] = m_sys.j_v[i, j, g]
                            js[g, i, (j+1), i, j] = m_sys.j_h[i, j, g]
                    for i in range(m_sys.n[0] - 1):
                        j = m_sys.n[1] - 1
                        js[g, i, j, i+1, j] = m_sys.j_v[i, j, g]
                        js[g, i+1, j, i, j] = m_sys.j_v[i, j, g]
                    for j in range(m_sys.n[1] - 1):
                        i = m_sys.n[0] - 1
                        js[g, i, j, i, (j+1)] = m_sys.j_h[i, j, g]
                        js[g, i, (j+1), i, j] = m_sys.j_h[i, j, g]
                    if m_sys.boundaries[1] == 0:
                        for i in range(m_sys.n[0]):
                            j = m_sys.n[1] - 1
                            js[g, i, j, i, 0] = m_sys.j_h[i, j, g]
                            js[g, i, 0, i, j] = m_sys.j_h[i, j, g]
                    if m_sys.boundaries[0] == 0:
                        for j in range(m_sys.n[1]):
                            i = m_sys.n[0] - 1
                            js[g, i, j, 0, j] = m_sys.j_v[i, j, g]
                            js[g, 0, j, i, j] = m_sys.j_v[i, j, g]
            
            elif(g<self.nbr_2Q_sys + self.nbr_2Q_anc):
                g_anc = g - self.nbr_2Q_sys
                #Ancilla js
                if(self.anc_fc):
                    for i in range(m_anc.n[0]):
                        for j in range(m_anc.n[1]):
                            for l in range(j+1, m_anc.n[1], 1):
                                js[g, m_sys.n[0] + i, j, m_sys.n[0]+i, l] = m_anc.j[i, j, i, l, g_anc]
                                js[g, m_sys.n[0] + i, l, m_sys.n[0]+i, j] = m_anc.j[i, j, i, l, g_anc]
                            for k in range(i+1, m_anc.n[0], 1):
                                for l in range(m_anc.n[1]):
                                    js[g, m_sys.n[0] + i, j, m_sys.n[0] + k, l] = m_anc.j[i, j, k, l, g_anc]
                                    js[g, m_sys.n[0] + k, l, m_sys.n[0] + i, j] = m_anc.j[i, j, k, l, g_anc]
                else:
                    for i in range(m_anc.n[0]-1):
                        for j in range(m_anc.n[1]-1):
                            js[g, m_sys.n[0] + i, j, m_sys.n[0] + i+1, j] = m_anc.j_v[i, j, g_anc]
                            js[g, m_sys.n[0] + i, j, m_sys.n[0] + i, j+1] = m_anc.j_h[i, j, g_anc]
                            js[g, m_sys.n[0] + i+1, j, m_sys.n[0] + i, j] = m_anc.j_v[i, j, g_anc]
                            js[g, m_sys.n[0] + i, j+1, m_sys.n[0] + i, j] = m_anc.j_h[i, j, g_anc]
                    for i in range(m_anc.n[0] - 1):
                        j = m_anc.n[1] - 1
                        js[g, m_sys.n[0] + i, j, m_sys.n[0] + i+1, j] = m_anc.j_v[i, j, g_anc]
                        js[g, m_sys.n[0] + i+1, j, m_sys.n[0] + i, j] = m_anc.j_v[i, j, g_anc]
                    for j in range(m_anc.n[1] - 1):
                        i = m_anc.n[0] - 1
                        js[g, m_sys.n[0] + i, j, m_sys.n[0] + i, j+1] = m_anc.j_h[i, j, g_anc]
                        js[g, m_sys.n[0] + i, j+1, m_sys.n[0] + i, j] = m_anc.j_h[i, j, g_anc]
                    if m_anc.boundaries[1] == 0:
                        for i in range(m_anc.n[0]):
                            j = m_anc.n[1] - 1
                            js[g, m_sys.n[0] + i, j, m_sys.n[0] + i, 0] = m_anc.j_h[i, j, g_anc]
                            js[g, m_sys.n[0] + i, 0, m_sys.n[0] + i, j] = m_anc.j_h[i, j, g_anc]
                    if m_anc.boundaries[0] == 0:
                        for j in range(m_anc.n[1]):
                            i = m_anc.n[0] - 1
                            js[g, m_sys.n[0] + i, j, m_sys.n[0], j] = m_anc.j_v[i, j, g_anc]
                            js[g, m_sys.n[0], j, m_sys.n[0] + i, j] = m_anc.j_v[i, j, g_anc]
            else:
                g_int = g - self.nbr_2Q_sys - self.nbr_2Q_anc
                #Interaction js
                for i in range(m_sys.n[0]):
                    for j in range(m_sys.n[1]):
                        js[g, i, j, m_sys.n[0] + i, j] = j_int[g_int, i, j]
                        js[g, m_sys.n[0] + i, j, i, j] = j_int[g_int, i, j]
        
        h = np.zeros(shape=(self.nbr_1Q, *n))
        for g in range(self.nbr_1Q):
            if(g<self.nbr_1Q_sys):
                #System hs
                for i in range(m_sys.n[0]):
                    for j in range(m_sys.n[1]):
                        h[g, i, j] = m_sys.h[i, j, g]
            else:
                g_anc = g - self.nbr_1Q_sys
                #Ancilla hs
                for i in range(m_anc.n[0]):
                    for j in range(m_anc.n[1]):
                        h[g, m_sys.n[0] + i, j] = m_anc.h[i, j, g_anc]
        
        return js, h
    
    def energy(self) -> np.ndarray:
        raise NotImplementedError('Cooling Energy not implemented, use expectation value of self.hamiltonian instead.') 
    
    def copy(self) -> CoolingNA:
        self_copy = CoolingNA( 
                self.m_sys,
                self.m_anc,
                self._two_q_gates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                self.j_int,
                self.t)

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
                "m_sys": self.m_sys,
                "m_anc": self.m_anc,
                "int_gates": self._two_q_gates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                "j_int": self.j_int,
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