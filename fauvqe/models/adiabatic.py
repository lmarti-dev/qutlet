from __future__ import annotations

import importlib
from typing import Tuple, Dict, Literal, Union
from numbers import Real

import numpy as np
import cirq

from fauvqe.models.spinModel_fc import SpinModelFC


class Adiabatic(SpinModelFC):
    """
       Adiabatic Sweep through fully connected Spin Models inherits SpinModelFC (-> is itself a SpinModelFC)
    """
    def __init__(self, 
                 H0: Union[SpinModelFC, SpinModel],
                 H1: Union[SpinModelFC, SpinModel],
                 sweep: lambda = None,
                 t: Real = 0,
                 T: Real = 1):
        """
        H0: initial Hamiltonian (t=0)
        H1: final Hamiltonian (t=T)
        t: Simulation time
        T: total time of adiabatic sweep
        """
        assert H0.qubittype == H1.qubittype, "Qubit types incompatible, received \nH0: {} \nand \nH1: {}".format(H0.qubittype, H1.qubittype)
        
        assert H0.n == H1.n, "Qubit numbers incompatible, received \nH0: {} \nand \nH1: {}".format(H0.n, H1.n)
        
        assert t >= 0 and t <= T, "Simulation time incompatible with adiabatic sweep time, received \nt= {} \nand \nT= {}".format(t, T)
        
        if(sweep is None):
            sweep = lambda time: time/T
        else:
            assert abs(sweep(0))<1e-13 and abs(sweep(T) - 1)<1e-13, "Handed sweep is not a switch function, instead sweep(0)= {} and sweep(T) = {}".format(sweep(0), sweep(T))
        
        self._H0 = H0
        self._H1 = H1
        self.T = T
        self._sweep = sweep
        
        ########################################################################
        
        super().__init__(H0.qubittype, 
                 np.array(H0.n),
                 #np.array([j_x, j_y, j_z]),
                 #np.array(h).reshape((3, n[0], n[1])),
                 #[lambda q1, q2: cirq.X(q1)*cirq.X(q2),
                 # lambda q1, q2: cirq.Y(q1)*cirq.Y(q2),
                 # lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)],
                 #[cirq.X, cirq.Y, cirq.Z],
                 t
        )
        self.energy_fields = ["X", "Y", "Z"]
    
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

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return [super().energy( self.j[:,:,:,:,0], self.h[:,:,0]), 
                super().energy( self.j[:,:,:,:,1], self.h[:,:,1]), 
                super().energy( self.j[:,:,:,:,2], self.h[:,:,2])]

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_x": self.j[:,:,:,:,0],
                "j_y": self.j[:,:,:,:,1],
                "j_z": self.j[:,:,:,:,2],
                "h_x": self.h[:,:,0],
                "h_y": self.h[:,:,1],
                "h_z": self.h[:,:,2],
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