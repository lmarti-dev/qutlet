from __future__ import annotations

import importlib
from typing import Tuple, Dict, Literal
from numbers import Real

import numpy as np
import cirq

from fauvqe.models.spinmodel import SpinModel

class Heisenberg(SpinModel):
    """
    2D Heisenberg class inherits SpinModel
    """
    def __init__(self, qubittype, n, j_x_v, j_x_h, j_y_v, j_y_h, j_z_v, j_z_h, h_x = None, h_y = None, h_z = None, t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j_x_v vertical j's for XX interaction
        j_x_h horizontal j's for XX interaction
        j_y_v vertical j's for YY interaction
        j_y_h horizontal j's for YY interaction
        j_z_v vertical j's for ZZ interaction
        j_z_h horizontal j's for ZZ interaction
        h_x  strength external field X
        h_y  strength external field Y
        h_z  strength external field Z
        """
        # convert all input to np array to be sure
        h = [h_x, h_y, h_z]
        for i in range(len(h)):
            if h[i] is None:
                h[i] = np.zeros((n[0], n[1]))
        super().__init__(qubittype, 
                 np.array(n),
                 [j_x_v, j_y_v, j_z_v],
                 [j_x_h, j_y_h, j_z_h],
                 np.array(h).reshape((3, n[0], n[1])),
                 [lambda q1, q2: cirq.X(q1)*cirq.X(q2),
                  lambda q1, q2: cirq.Y(q1)*cirq.Y(q2),
                  lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)],
                 [cirq.X, cirq.Y, cirq.Z],
                 t
        )
        self.energy_fields = ["X", "Y", "Z"]
    
    def copy(self) -> Heisenberg:
        print(self.j_v)
        self_copy = Heisenberg( self.qubittype,
                self.n,
                self.j_v[:,:,0],
                self.j_h[:,:,0],
                self.j_v[:,:,1],
                self.j_h[:,:,1],
                self.j_v[:,:,2],
                self.j_h[:,:,2],
                self.h[:,:,0],
                self.h[:,:,1],
                self.h[:,:,2],
                self.t )
        
        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy._hamiltonian = self._hamiltonian.copy()
        
        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return [super().energy( self.j_v[:,:,0], self.j_h[:,:,0], self.h[:,:,0]), 
                super().energy( self.j_v[:,:,1], self.j_h[:,:,1], self.h[:,:,1]), 
                super().energy( self.j_v[:,:,2], self.j_h[:,:,2], self.h[:,:,2])]

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_x_v": self.j_v[:,:,0],
                "j_x_h": self.j_h[:,:,0],
                "j_y_v": self.j_v[:,:,1],
                "j_y_h": self.j_h[:,:,1],
                "j_z_v": self.j_v[:,:,2],
                "j_z_h": self.j_h[:,:,2],
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