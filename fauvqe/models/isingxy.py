from __future__ import annotations

import importlib
from typing import Tuple, Dict, Literal
from numbers import Real

import numpy as np
import cirq

from fauvqe.models.spinModel import SpinModel


class IsingXY(SpinModel):
    """
    2D Ising XY class inherits SpinModel
    """
    def __init__(self, qubittype, n, j_y_v, j_y_h, j_z_v, j_z_h, h, field: Literal["Z", "X"] = "X", t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j_y_v vertical j's for YY interaction
        j_y_h horizontal j's for YY interaction
        j_z_v vertical j's for ZZ interaction
        j_z_h horizontal j's for ZZ interaction
        h  strength external field
        field: basis of external field X or Z
        """
        # convert all input to np array to be sure
        if(field == "X"):
            SingleQubitGate = [cirq.X,]
            self.energy_fields = ["Y", "Z", "X"]
        elif(field == "Z"):
            SingleQubitGate = [cirq.Z,]
            self.energy_fields = ["Y", "Z"]
        else:
            assert False, "Incompatible field name, expected: 'X' or 'Z', received: " + str(field)
        super().__init__(qubittype, 
                 np.array(n),
                 [j_y_v, j_z_v],
                 [j_y_h, j_z_h],
                 [h,],
                 [lambda q1, q2: cirq.Y(q1)*cirq.Y(q2),
                  lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)],
                 SingleQubitGate,
                 t
        )
        self.field = field

    def copy(self) -> IsingXY:
        self_copy = IsingXY( self.qubittype,
                self.n,
                self.j_v[:,:,0],
                self.j_h[:,:,0],
                self.j_v[:,:,1],
                self.j_h[:,:,1],
                self.h[:,:,0],
                self.field,
                self.t )

        self_copy.circuit = self.circuit.copy()
        self_copy.circuit_param = self.circuit_param.copy()
        self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy.hamiltonian = self.hamiltonian.copy()

        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        if(self.field == "X"):
            return [super().energy( self.j_v[:,:,0], self.j_h[:,:,0], np.zeros(self.h.shape)), 
                    super().energy( self.j_v[:,:,1], self.j_h[:,:,1], np.zeros(self.h.shape)), 
                    super().energy( np.zeros(self.j_v[:, :, 0].shape), np.zeros(self.j_h[:, :, 0].shape), self.h[:,:,0])]
        else:
            return [super().energy( self.j_v[:,:,0], self.j_h[:,:,0], np.zeros(self.h.shape)),
                    super().energy( self.j_v[:,:,1], self.j_h[:,:,1], self.h[:,:,0])]

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_y_v": self.j_v[:,:,0],
                "j_y_h": self.j_h[:,:,0],
                "j_z_v": self.j_v[:,:,1],
                "j_z_h": self.j_h[:,:,1],
                "h": self.h[:,:,0],
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