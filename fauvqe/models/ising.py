from __future__ import annotations

import importlib
from typing import Tuple, Dict, Literal
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.spinmodel import SpinModel


class Ising(SpinModel):
    """
    2D Ising class inherits AbstractModel
    is mother of different quantum circuit methods
    """
    
    def __init__(self, qubittype, n, j_v = None, j_h = None, h = None, field: Literal["Z", "X"] = "X", t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j_v vertical j's
        j_h horizontal j's
        h  strength external field
        field: basis of external field X or Z
        """
        self.__name__ = "IsingModel"

        # convert all input to np array to be sure
        if j_v is None:
            j_v = np.zeros((n[0], n[1]))
        if j_h is None:
            j_h = np.zeros((n[0], n[1]))
        if h is None:
            h = np.zeros((n[0], n[1]))
        if(field == "X"):
            one_q_gate = [cirq.X]
            self.energy_fields = ["Z", "X"]
        elif(field == "Z"):
            one_q_gate = [cirq.Z]
            self.energy_fields = ["Z"]
        else:
            assert False, "Incompatible field name, expected: 'X' or 'Z', received: " + str(field)
        super().__init__(
                    qubittype, 
                    np.array(n),
                    [j_v],
                    [j_h],
                    [h],
                    [lambda q1, q2: cirq.Z(q1)*cirq.Z(q2)],
                    one_q_gate,
                    t
                )
        self.field = field

    def copy(self) -> Ising:
        self_copy = Ising( self.qubittype,
                self.n,
                self.j_v[:,:,0],
                self.j_h[:,:,0],
                self.h[:,:,0],
                self.field,
                self.t )

        self_copy.circuit = self.circuit.copy()
        if self.circuit_param is not None: self_copy.circuit_param = self.circuit_param.copy()
        if self.circuit_param_values is not None: self_copy.circuit_param_values = self.circuit_param_values.copy()
        self_copy._hamiltonian = self._hamiltonian.copy()

        if self.eig_val is not None: self_copy.eig_val = self.eig_val.copy()
        if self.eig_vec is not None: self_copy.eig_vec = self.eig_vec.copy()
        if self._Ut is not None: self_copy._Ut = self._Ut.copy()

        return self_copy

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        if(self.field == "X"):
            return [super().energy( self.j_v, self.j_h, np.zeros(self.h.shape)), 
                    super().energy( np.zeros(self.j_v[:, :, 0].shape), np.zeros(self.j_h[:, :, 0].shape), self.h)]
        else:
            return [super().energy( self.j_v, self.j_h, self.h)]

    def get_spin_vm(self, wf):
        assert np.size(self.n) == 2, "Expect 2D qubit grid"
        # probability from wf
        prob = abs(wf * np.conj(wf))

        # cumulative probability
        n_temp = round(np.log2(wf.shape[0]))
        com_prob = np.zeros(n_temp)
        # now sum it
        # this is a potential openmp sum; loop over com_prob, use index arrays to select correct

        for i in np.arange(n_temp):
            # com_prob[i] = sum wf over index array all in np
            # maybe write on
            # does not quite work so do stupid version with for loop
            # declaring cpython types can maybe help,
            # Bad due to nested for for if instead of numpy, but not straight forward
            for j in np.arange(2 ** n_temp):
                # np.binary_repr(3, width=4) use as mask
                if np.binary_repr(j, width=n_temp)[i] == "1":
                    com_prob[i] += prob[j]
        # This is for qubits:
        # {(i1, i2): com_prob[i2 + i1*q4.n[1]] for i1 in np.arange(q4.n[0]) for i2 in np.arange(q4.n[1])}
        # But we want for spins:
        return {
            (i0, i1): 2 * com_prob[i1 + i0 * self.n[1]] - 1
            for i0 in np.arange(self.n[0])
            for i1 in np.arange(self.n[1])
        }

    def print_spin(self, wf):
        """
        Currently does not work due to Cirq update...

        For cirq. heatmap see example:
        https://github.com/quantumlib/Cirq/blob/master/examples/bristlecone_heatmap_example.py
        https://github.com/quantumlib/Cirq/blob/master/examples/heatmaps.py
        https://github.com/quantumlib/Cirq/blob/master/cirq-core/cirq/vis/heatmap_test.py
        value_map = {
            (qubit.row, qubit.col): np.random.random() for qubit in cirq.google.Bristlecone.qubits
        }
        heatmap = cirq.Heatmap(value_map)
        heatmap.plot()

        This is hard to test, but self.get_spin_vm(wf) is covered
        Possibly add test similar to cirq/vis/heatmap_test.py

        Further: add colour scale
        """
        value_map = self.get_spin_vm(wf)
        # Create heatmap object
        heatmap = cirq.Heatmap(value_map)
        # Plot heatmap
        heatmap.plot()

    def energy_pfeuty_sol(self):
        """
        Function that returns analytic solution for ground state energy
        of 1D TFIM as described inj Pfeuty, ANNALS OF PHYSICS: 57, 79-90 (1970)
        Currently this ONLY WORKS FOR PERIODIC BOUNDARIES

        First assert if following conditions are met:
            -The given system is 1D
            -all h's have the same value
            -all J's have the same value, independant of which one is the
                'used' direction

        Then:
            - Calculate \Lambda_k
                For numeric reasons include h in \Lambda_k
            - Return E/N = - h* sum \Lambda_k/N
        """
        assert self.n[0] * self.n[1] == np.max(
            self.n
        ), "Ising class error, given system dimensions n = {} are not 1D".format(self.n)
        assert np.min(self.h) == np.max(
            self.h
        ), "Ising class error, external field h = {} is not the same for all spins".format(self.h)
        
        # Use initial parameter to catch empty array
        assert (
            np.min(self.j_h, initial=np.finfo(np.float_).max)
            == np.max(self.j_h, initial=np.finfo(np.float_).min)
        ) or (
            np.size(self.j_h) == 0
        ), "Ising class error, interaction strength j_h = {} is not the same for all spins. max: {} , min: {}".format(
            self.j_h,
            np.min(self.j_h, initial=np.finfo(np.float_).max),
            np.max(self.j_h, initial=np.finfo(np.float_).min),
        )

        # Use initial parameter to catch empty array
        assert (
            np.min(self.j_v, initial=np.finfo(np.float_).max)
            == np.max(self.j_v, initial=np.finfo(np.float_).min)
        ) or (
            np.size(self.j_v) == 0
        ), "Ising class error, interaction strength j_v = {} is not the same for all spins. max: {} , min: {}".format(
            self.j_v,
            np.min(self.j_v, initial=np.finfo(np.float_).max),
            np.max(self.j_v, initial=np.finfo(np.float_).min),
        )

        lambda_k = self._get_lambda_k()
        return -np.sum(lambda_k) /(self.n[0]*self.n[1])

    def _get_lambda_k(self):
        """
        Helper function for energy_pfeuty_sol()
        Not intended for external call
        """
        _n = self.n[0] * self.n[1]
        _k = (
            2 * np.pi * np.arange(start=-(_n - np.mod(_n, 2)) / 2, stop=_n / 2 , step=1, dtype=np.complex128) / _n
        )

        if self.j_h.size > 0:
            _j = self.j_h[0][0]
        else:
            _j = self.j_v[0][0]

        return np.sqrt(self.h[0][0][0] ** 2 + _j ** 2 - (2 * _j) * self.h[0][0][0] * np.cos(_k), dtype=np.complex128)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "qubittype": self.qubittype,
                "n": self.n,
                "j_v": self.j_v[:,:,0],
                "j_h": self.j_h[:,:,0],
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