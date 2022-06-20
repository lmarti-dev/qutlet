from __future__ import annotations

import importlib
from typing import Tuple, Dict, List
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.spinModel_fc import SpinModelFC
from fauvqe.models.abstractmodel import AbstractModel

class CoolingModel(SpinModelFC):
    """
    CoolingModel class
    
    Class for cooling structure using n system qubits and either a copy of the system or a single added row as ancilla qubit; inherits SpinModelFC
    
    Parameters
    ----------
    m_sys: AbstractModel, System Model
    m_anc: AbstractModel, Ancilla Model
    int_gates: List[cirq.PauliSum], List of gates used for the interaction between system and ancilla qubits
    j_int: np.array, interaction coefficients used on the interaction gates int_gates
    t: Real, simulation time
    
    Methods
    ----------
    _get_int_index(self, i:int): int
        Returns
        ---------
        int:
            Index which determines the ancilla qubit connected to the ith row of the system
    
    _combine_jh(self): List[np.array]
        Returns
        ---------
        List[np.array]:
            Arrays defining interaction js and field strengths h
    
    _set_j_int(self, j_int): void
        Returns
        ---------
        void
    
    _set_h_anc(self, h_anc): void
        Returns
        ---------
        void
    """
    
    def __init__(self, 
                 m_sys: AbstractModel,
                 m_anc: AbstractModel,
                 int_gates: List[cirq.PauliSum],
                 j_int: np.array,
                 t: Real = 0):
        """
        m_sys: model defining the system of interest
        m_anc: model defining the ancilla system
        int_gates: a cirq.PauliSum defining the interaction gates between system and ancillas including interaction constants
        j_int: an array encoding the interaction constants between ancillas and system
        t: Simulation Time
        """
        assert m_sys.qubittype == m_anc.qubittype, "Incompatible Qubittypes, System: {}, Ancilla: {}".format(
            m_sys.qubittype, m_anc.qubittype
        )
        j_int = np.array(j_int)
        assert j_int.shape == (len(int_gates), *m_sys.n ), "Wrong shape of j_int, received: {} expected: {}".format(
            j_int.shape, (len(int_gates), *m_sys.n)
        )
        if(m_anc.n[0] == m_sys.n[0] and m_anc.n[1] == m_sys.n[1]):
            n = [2*m_sys.n[0], m_sys.n[1]]
            self.cooling_type = "NA"
            self._get_int_index = self._get_int_index_na
        elif(m_anc.n[0] == 1 and m_anc.n[1] == m_sys.n[1]):
            n = [m_sys.n[0] + 1, m_sys.n[1]]
            self.cooling_type = "1A"
            self._get_int_index = self._get_int_index_1a
        else:
            assert False, "Could not instantiate system doubled ancilla cooling or single qubit ancilla cooling with ancilla model of size {}".format(
                m_anc.n
            )
        
        self.nbr_2Q_sys = len(m_sys._TwoQubitGates)
        self.nbr_2Q_anc = len(m_anc._TwoQubitGates)
        self.nbr_2Q_int = len(int_gates)
        self.nbr_2Q = self.nbr_2Q_sys + self.nbr_2Q_anc + self.nbr_2Q_int
        
        self.nbr_1Q_sys = len(m_sys._SingleQubitGates)
        self.nbr_1Q_anc = len(m_anc._SingleQubitGates)
        self.nbr_1Q = self.nbr_1Q_sys + self.nbr_1Q_anc
        
        self.sys_fc = issubclass(m_sys.__class__, SpinModelFC)
        self.anc_fc = issubclass(m_anc.__class__, SpinModelFC)
        
        _TwoQubitGates = [*m_sys._TwoQubitGates, *m_anc._TwoQubitGates, *int_gates]
        _SingleQubitGates = [*m_sys._SingleQubitGates, *m_anc._SingleQubitGates]
        
        self.m_sys = m_sys
        self.m_anc = m_anc
        self.j_int = j_int
        
        j, h = self._combine_jh()
        
        super().__init__( 
            m_sys.qubittype,
            np.array(n),
            j,
            h,
            _TwoQubitGates,
            _SingleQubitGates,
            t
        )
    
    def _get_int_index_na(self, i: int) -> int:
        return self.m_sys.n[0] + i
    
    def _get_int_index_1a(self, i: int) -> int:
        return self.m_sys.n[0]
    
    def _combine_jh(self) -> List[np.array]:
        """Combine interaction and field strengths from system and ancilla qubits to get the new interaction graph

        Parameters
        ----------
        self
        
        Returns
        -------
        js: np.array
        h: np.array
        """
        m_sys = self.m_sys
        m_anc = self.m_anc
        j_int = self.j_int
        if(self.cooling_type == "NA"):
            n = [2*m_sys.n[0], m_sys.n[1]]
        else:
            n = [m_sys.n[0] + 1, m_sys.n[1]]
        g_int = 0
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
                        js[g, i, j, self._get_int_index(i), j] = j_int[g_int, i, j]
                        js[g, self._get_int_index(i), j, i, j] = j_int[g_int, i, j]
        
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
    
    def _set_j_int(self, j_int: np.array) -> None:
        """Sets the new interaction constants j_int and recombines the whole interaction graph.
            To be called when j_int shall be changed after already having initialized the object.
        
        Parameters
        ----------
        self
        j_int: np.array
        
        Returns
        -------
        void
        """
        self.j_int = j_int
        for g in range(self.nbr_2Q_sys + self.nbr_2Q_anc, self.nbr_2Q):
            g_int = g - self.nbr_2Q_sys - self.nbr_2Q_anc
            #Interaction js
            for i in range(self.m_sys.n[0]):
                for j in range(self.m_sys.n[1]):
                    self.j[i, j, self._get_int_index(i), j, g] = j_int[g_int, i, j]
                    self.j[self._get_int_index(i), j, i, j, g] = j_int[g_int, i, j]
    
    def _set_h_anc(self, h_anc: np.array) -> None:
        """Sets the new ancilla field strengts h_anc and recombines the whole interaction graph.
            To be called when h_anc shall be changed after already having initialized the object.
            
        Parameters
        ----------
        self
        h_anc: np.array
        
        Returns
        -------
        void
        """
        for g in range(self.nbr_1Q_sys, self.nbr_1Q):
            g_anc = g - self.nbr_1Q_sys
            for i in range(self.m_anc.n[0]):
                for j in range(self.m_anc.n[1]):
                    self.h[self.m_sys.n[0]+i, j, g] = h_anc[i, j, g_anc]
    
    def energy(self) -> np.ndarray:
        raise NotImplementedError('Cooling Energy not implemented, use expectation value of self.hamiltonian instead.') 
    
    def copy(self) -> CoolingModel:
        self_copy = CoolingModel(
                self.m_sys,
                self.m_anc,
                self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                self.j_int,
                self.t)
        
        self_copy.circuit = self.circuit.copy()
        if self.circuit_param is not None: self_copy.circuit_param = self.circuit_param.copy()
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
                "int_gates": self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
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