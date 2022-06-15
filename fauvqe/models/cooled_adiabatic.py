from __future__ import annotations

from collections.abc import Callable
from typing import Dict, Union
from numbers import Real
from scipy.integrate import quad
import scipy

import numpy as np
import cirq

from fauvqe.models.coolingmodel import CoolingModel
from fauvqe.models.adiabatic import Adiabatic
import fauvqe


class CooledAdiabatic(CoolingModel):
    """
       Cooling assisted Adiabatic Sweep through fully connected Spin Models inherits SpinModelFC (-> is itself a SpinModelFC)
    """
    def __init__(self, 
                 H0: Union[SpinModelFC, SpinModel],
                 H1: Union[SpinModelFC, SpinModel],
                 m_anc: AbstractModel,
                 int_gates: List[cirq.PauliSum],
                 j_int: np.array,
                 sweep: Callable = None,
                 t: Real = 0,
                 T: Real = 1
                ):
        """
        H0: initial Hamiltonian (t=0)
        H1: final Hamiltonian (t=T)
        t: Simulation time
        T: total time of adiabatic sweep
        m_anc: model defining the ancilla system
        int_gates: a cirq.PauliSum defining the interaction gates between system and ancillas including interaction constants
        j_int: an array encoding the interaction constants between ancillas and system
        """
        m_sys = Adiabatic(H0,
                          H1,
                          sweep,
                          t,
                          T)
        
        self._Uts: Optional[np.ndarray] = None
        
        super().__init__(m_sys,
                 m_anc,
                 int_gates,
                 j_int,
                 t)
    
    def copy(self) -> CooledAdiabatic:
        self_copy = CooledAdiabatic( 
                self.m_sys._H0,
                self.m_sys._H1,
                self.m_anc,
                self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                self.j_int,
                self.m_sys._sweep,
                self.m_sys.t,
                self.m_sys.T
        )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy.hamiltonian = self.hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy
    
    #Overrides CoolingModel's function
    def _set_hamiltonian(self, reset: bool = True) -> None:
        if reset:
            self.hamiltonian = cirq.PauliSum()
        self.hamiltonian += self._get_hamiltonian_at_time(self.t)
    
    def _get_hamiltonian_at_time(self, time: Real):
        """
        Append or Reset Hamiltonian; Combine Hamiltonians:
            (1 - sweep(t)) * H0 + sweep(t) * H1
            + m_anc.hamiltonian
            + interaction term
        
        Parameters
        ----------
        self
        reset: bool, indicates whether to reset or append Hamiltonian
        
        Returns
        -------
        void 
        """
        ham = cirq.PauliSum()
        
        self.m_sys.t = time
        self.m_sys._set_hamiltonian()
        self.m_anc.t = time
        self.m_anc._set_hamiltonian()
        
        sys_rows = self.m_sys.n[0]
        new_qubits = [self.qubits[i][j] for i in range(sys_rows, self.n[0], 1) for j in range(self.n[1])]
        ham = self.m_sys.hamiltonian + self.m_anc.hamiltonian.with_qubits(*new_qubits)
        int_gates = self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q]
        for g in range(len(int_gates)):
            for i in range(self.m_sys.n[0]):
                for j in range(self.m_sys.n[1]):
                    if(self.j_int[g][i][j] != 0):
                        if self.cooling_type == "NA":
                            ham -= self.j_int[g][i][j] * cirq.PauliSum.from_pauli_strings(int_gates[g](self.qubits[i][j], self.qubits[i+sys_rows][j]))
                        else:
                            ham -= self.j_int[g][i][j] * cirq.PauliSum.from_pauli_strings(int_gates[g](self.qubits[i][j], self.qubits[sys_rows][j]))
        
        return ham
    
    #Only Trotterization with adiabatic assumption possible -> alternative Integrate Schrödinger equation directly
    def set_Uts(self, trotter_steps: int = 0) -> None:
        if(trotter_steps == 0):
            trotter_steps = int(self.m_sys.T)
        delta_t = self.m_sys.T / trotter_steps
        
        self._Uts = []
        for m in range(trotter_steps):
            hamiltonian = self._get_hamiltonian_at_time(m*delta_t)
            #print(hamiltonian)
            hamiltonian = hamiltonian.matrix()
            eig_val, eig_vec =  np.linalg.eigh(hamiltonian)
            self._Uts.append( 
                np.matmul(np.matmul(eig_vec, np.diag( np.exp( -1j * delta_t * eig_val ) ), dtype = np.complex64), eig_vec.conjugate().transpose())
            )
    
    def _get_default_trotter_steps(self, nbr_resets):
        return int(self.m_sys.T) - (int(self.m_sys.T) % nbr_resets) + nbr_resets

    def perform_sweep(self, nbr_resets: int = None) -> np.ndarray:
        #Set number of resets
        if nbr_resets is None:
            dt = 2 * np.pi / self.m_sys._get_minimal_energy_gap()
            nbr_resets = int(self.m_sys.T) / dt
        #Set Uts for sweep
        if self._Uts is None or (len(self._Uts) % nbr_resets != 0 ):
            self.set_Uts(self._get_default_trotter_steps(nbr_resets))
        steps = len(self._Uts)
        # Do the sweep with intermediate resets
        
        final = 0
        return final
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "H0": self.m_sys._H0,
                "H1": self.m_sys._H1,
                "m_anc": self.m_anc,
                "int_gates": self._TwoQubitGates[self.nbr_2Q_sys + self.nbr_2Q_anc:self.nbr_2Q],
                "j_int": self.j_int,
                "sweep": self.m_sys._sweep,
                "t": self.m_sys.t,
                "T": self.m_sys.T
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