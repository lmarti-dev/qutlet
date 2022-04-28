from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Tuple, Dict, Literal, Union
from numbers import Real
from scipy.integrate import quad

import numpy as np
import cirq

from fauvqe.models.spinModel_fc import SpinModelFC
import fauvqe


class Adiabatic(SpinModelFC):
    """
       Adiabatic Sweep through fully connected Spin Models inherits SpinModelFC (-> is itself a SpinModelFC)
    """
    def __init__(self, 
                 H0: Union[SpinModelFC, SpinModel],
                 H1: Union[SpinModelFC, SpinModel],
                 sweep: Callable[Real] = None,
                 t: Real = 0,
                 T: Real = 1):
        """
        H0: initial Hamiltonian (t=0)
        H1: final Hamiltonian (t=T)
        t: Simulation time
        T: total time of adiabatic sweep
        """
        assert H0.qubittype == H1.qubittype, "Qubit types incompatible, received \nH0: {} \nand \nH1: {}".format(H0.qubittype, H1.qubittype)
        
        assert (H0.n == H1.n).all(), "Qubit numbers incompatible, received \nH0: {} \nand \nH1: {}".format(H0.n, H1.n)
        
        assert t >= 0 and t <= T, "Simulation time incompatible with adiabatic sweep time, received \nt= {} \nand \nT= {}".format(t, T)
        
        if(sweep is None):
            sweep = lambda time: time/T
        else:
            assert abs(sweep(0))<1e-13 and abs(sweep(T) - 1)<1e-13, "Handed sweep is not a switch function, instead sweep(0)= {} and sweep(T) = {}".format(sweep(0), sweep(T))
        
        self._H0 = H0
        self._H1 = H1
        
        if(isinstance(H0, fauvqe.SpinModel)):
            #self._H0 = SpinModelFC.toFC(H0)
            self._H0.j = SpinModelFC.toFC(H0)
        
        if(isinstance(H1, fauvqe.SpinModel)):
            #self._H1 = SpinModelFC.toFC(H1)
            self._H1.j = SpinModelFC.toFC(H1)
        
        self.t = t
        self.T = T
        self._sweep = sweep
        
        j_tot, h_tot, TwoQubitGates, SingleQubitGates = self.get_interactions()
        
        super().__init__(H0.qubittype, 
                np.array(H0.n),
                np.array(j_tot),
                np.array(h_tot),
                TwoQubitGates,
                SingleQubitGates,
                t
        )
    
    def get_interactions(self) -> List[list]:
        self.energy_fields = [*self._H0.energy_fields, *self._H1.energy_fields]
        
        l = self._sweep(self.t)
        
        j0 = np.transpose(self._H0.j, (4, 0, 1, 2, 3))
        j1 = np.transpose(self._H1.j, (4, 0, 1, 2, 3))
        h0 = np.transpose(self._H0.h, (2, 0, 1))
        h1 = np.transpose(self._H1.h, (2, 0, 1))
        
        j_tot = np.array([*(1-l)*j0, *l*j1])
        h_tot = np.array([*(1-l)*h0, *l*h1])
        
        TwoQubitGates = [*self._H0._TwoQubitGates, *self._H1._TwoQubitGates]
        SingleQubitGates = [*self._H0._SingleQubitGates, *self._H1._SingleQubitGates]
        
        return j_tot, h_tot, TwoQubitGates, SingleQubitGates
    
    def copy(self) -> Adiabatic:
        self_copy = Adiabatic( 
                self._H0,
                self._H1,
                self._sweep,
                self.t,
                self.T 
        )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy.hamiltonian = self.hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy
    
    #Overrides SpinModelFC's function
    def _set_hamiltonian(self, reset: bool = True) -> None:
        """
        Append or Reset Hamiltonian; Combine Hamiltonians:
            (1 - sweep(t)) * H0 + sweep(t) * H1
        
        Parameters
        ----------
        self
        reset: bool, indicates whether to reset or append Hamiltonian
        
        Returns
        -------
        void 
        """
        if reset:
            self.hamiltonian = cirq.PauliSum()
        
        self.hamiltonian = (1-self._sweep(self.t)) * self._H0.hamiltonian + self._sweep(self.t) * self._H1.hamiltonian
    
    #Overrides SpinModelFC's function
    def set_Ut(self):
        _n = np.size(self.qubits)
        _N = 2**(_n)
        
        if self.t == 0:
            self._Ut = np.identity(_N)
            return True
        
        sweep_integrated, error = quad(self._sweep, 0, self.t)
        if(error > 1e-13):
            print('WARNING: Numerical integration error: {}'.format(error))
        
        hamiltonian_integrated = ((self.t - sweep_integrated) * self._H0.hamiltonian + sweep_integrated * self._H1.hamiltonian).matrix()
        
        eig_val, eig_vec =  np.linalg.eigh(hamiltonian_integrated)
        
        self._Ut = eig_vec @ np.diag( np.exp( -1j * eig_val ) ) @ eig_vec.conjugate().transpose()
    
    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return [*((1 - self._sweep(self.t)) * np.array( self._H0.energy())),
                  *(self._sweep(self.t) * np.array(self._H1.energy()))]

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "H0": self._H0,
                "H1": self._H1,
                "sweep": self._sweep,
                "t": self.t,
                "T": self.T
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