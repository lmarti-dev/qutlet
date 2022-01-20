from __future__ import annotations

import importlib
from typing import Tuple, Dict, Literal
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.abstractmodel import AbstractModel



class IsingXY(SpinModel):
    """
    2D Ising class inherits SpinModel
    is mother of different quantum circuit methods
    """
    basics  = importlib.import_module("fauvqe.models.circuits.basics")
    hea  = importlib.import_module("fauvqe.models.circuits.hea")
    qaoa = importlib.import_module("fauvqe.models.circuits.qaoa")

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
            one_q_gate = [cirq.X,]
        elif(field == "Z"):
            one_q_gate = [cirq.Z,]
        else:
            assert False, "Incompatible field name, expected: 'X' or 'Z', received: " + str(field)
        super().__init__(qubittype, 
                 np.array(n),
                 [j_y_v, j_z_v],
                 [j_y_h, j_z_h],
                 [h,],
                 [cirq.YY, cirq.ZZ],
                 one_q_gate,
                 t
        )
        self.j_y_v = j_y_v
        self.j_y_h = j_y_h
        self.j_z_v = j_z_v
        self.j_z_h = j_z_h
        self.h = h
        self.field = field

    def copy(self) -> IsingXY:
        self_copy = IsingXY( self.qubittype,
                self.n,
                self.j_y_v,
                self.j_y_h,
                self.j_z_v,
                self.j_z_h,
                self.h,
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
        raise NotImplementedError()

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_y_v": self.j_y_v,
                "j_y_h": self.j_y_h,
                "j_z_v": self.j_z_v,
                "j_z_h": self.j_z_h,
                "h": self.h,
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