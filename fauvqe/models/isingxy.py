from __future__ import annotations

import importlib
from typing import Tuple, Dict, Literal
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.abstractmodel import AbstractModel



class IsingXY(AbstractModel):
    """
    2D Ising class inherits AbstractModel
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
        super().__init__(qubittype, np.array(n))
        self.circuit_param = None
        self.circuit_param_values = np.array([])
        self._set_jh(j_y_v, j_y_h, j_z_v, j_z_h, h)
        self.field = field
        self._set_hamiltonian()
        super().set_simulator()
        self.t = t

    def copy(self) -> Ising:
        self_copy = Ising( self.qubittype,
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

    def _set_jh(self, j_y_v, j_y_h, j_z_v, j_z_h, h):
        # convert input to numpy array to be sure
        j_y_v = np.array(j_y_v)
        j_z_v = np.array(j_z_v)
        # J vertical needs one row/horizontal line less
        # NEED FOR IMPROVEMENT
        assert (j_y_v.shape == (self.n - np.array((1, 0)))).all() or (
            j_y_v.shape == self.n
        ).all(), "Error in Ising._set_jh(): j_y_v.shape != n - {{ (1,0), (0,0)}}, {} != {}".format(
            j_y_v.shape, (self.n - np.array((1, 0)))
        )
        self.j_y_v = j_y_v
        
        assert (j_z_v.shape == (self.n - np.array((1, 0)))).all() or (
            j_z_v.shape == self.n
        ).all(), "Error in Ising._set_jh(): j_z_v.shape != n - {{ (1,0), (0,0)}}, {} != {}".format(
            j_z_v.shape, (self.n - np.array((1, 0)))
        )
        self.j_z_v = j_z_v
        
        # convert input to numpy array to be sure
        j_y_h = np.array(j_y_h)
        j_z_h = np.array(j_z_h)
        # J horizontal needs one column/vertical line less#
        # NEED FOR IMPROVEMENT
        assert (j_y_h.shape == (self.n - np.array((0, 1)))).all() or (
            j_y_h.shape == self.n
        ).all(), "Error in Ising._set_jh(): j_y_h.shape != n - {{ (0,1), (0,0)}}, {} != {}".format(
            j_y_h.shape, (self.n - np.array((0, 1)))
        )
        self.j_y_h = j_y_h
        
        assert (j_z_h.shape == (self.n - np.array((0, 1)))).all() or (
            j_z_h.shape == self.n
        ).all(), "Error in Ising._set_jh(): j_z_h.shape != n - {{ (0,1), (0,0)}}, {} != {}".format(
            j_z_h.shape, (self.n - np.array((0, 1)))
        )
        self.j_z_h = j_z_h

        # Set boundaries:
        self.boundaries = np.array((self.n[0] - j_y_v.shape[0], self.n[1] - j_y_h.shape[1]))
        boundaries_z = np.array((self.n[0] - j_z_v.shape[0], self.n[1] - j_z_h.shape[1]))
        assert (self.boundaries == boundaries_z).all(), 'Inconsistent boundaries for YY and ZZ term'
        
        # convert input to numpy array to be sure
        h = np.array(h)
        assert (
            h.shape == self.n
        ).all(), "Error in Ising._set_jh():: h.shape != n, {} != {}".format(h.shape, self.n)
        self.h = h

    def _set_hamiltonian(self, reset: bool = True):
        """
            Append or Reset Hamiltonian

            Create a cirq.PauliSum object fitting to j_y_v, j_y_h, j_z_v, j_z_h, h  
        """
        if reset:
            self.hamiltonian = cirq.PauliSum()
        
        #Conversion currently necessary as numpy type * cirq.PauliSum fails
        j_y_v = self.j_y_v.tolist()
        j_y_h = self.j_y_h.tolist()
        j_z_v = self.j_z_v.tolist()
        j_z_h = self.j_z_h.tolist()
        h = self.h.tolist()
        
        #print(self.n)
        # 1. Sum over inner bounds
        for i in range(self.n[0] - 1):
            for j in range(self.n[1] - 1):
                #print("i: \t{}, j: \t{}".format(i,j))
                self.hamiltonian -= j_y_v[i][j]*cirq.Y(self.qubits[i][j])*cirq.Y(self.qubits[i+1][j])
                self.hamiltonian -= j_y_h[i][j]*cirq.Y(self.qubits[i][j])*cirq.Y(self.qubits[i][j+1])
                self.hamiltonian -= j_z_v[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i+1][j])
                self.hamiltonian -= j_z_h[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i][j+1])

        for i in range(self.n[0] - 1):
            j = self.n[1] - 1
            self.hamiltonian -= j_y_v[i][j]*cirq.Y(self.qubits[i][j])*cirq.Y(self.qubits[i+1][j])
            self.hamiltonian -= j_z_v[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i+1][j])

        for j in range(self.n[1] - 1):
            i = self.n[0] - 1
            self.hamiltonian -= j_y_h[i][j]*cirq.Y(self.qubits[i][j])*cirq.Y(self.qubits[i][j+1])
            self.hamiltonian -= j_z_h[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i][j+1])
        
        #2. Sum periodic boundaries
        if self.boundaries[1] == 0:
            for i in range(self.n[0]):
                j = self.n[1] - 1
                self.hamiltonian -= j_y_h[i][j]*cirq.Y(self.qubits[i][j])*cirq.Y(self.qubits[i][0])
                self.hamiltonian -= j_z_h[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[i][0])

        if self.boundaries[0] == 0:
            for j in range(self.n[1]):
                i = self.n[0] - 1
                self.hamiltonian -= j_y_v[i][j]*cirq.Y(self.qubits[i][j])*cirq.Y(self.qubits[0][j])
                self.hamiltonian -= j_z_v[i][j]*cirq.Z(self.qubits[i][j])*cirq.Z(self.qubits[0][j])
        
        # 3. Add external field
        if self.field == "X":
            field_gate = cirq.X
        elif self.field == "Z":
            field_gate = cirq.Z

        for i in range(self.n[0]):
            for j in range(self.n[1]):
                self.hamiltonian -= h[i][j]*field_gate(self.qubits[i][j])

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def set_circuit(self, qalgorithm, options: dict = {}):
        """
        Adds custom circuit to self.circuit (default)

        Args:
            qalgorithm : quantum algorithm option
            param:
                hand over parameter to individual circuit method; e.g. qaoa
                reset circuit (-> self. circuit = cirq. Circuit())

        Returns/Sets:
            circuit symp.Symbols array
            start parameters for circuit_parametrisation values; possibly at random or in call

        -AssertionError if circuit method does not exists
        -AssertionErrors for wrong parameter hand-over in individual circuit method itself.

        maybe use keyword arguments **parm later

        Need to generalise beta, gamma, beta_values, gamma_values to:

        obj.circuit_param           %these are the sympy.Symbols
        obj.circuit_param_values    %these are the sympy.Symbols values

        What to do with further circuit parameters like p?

        for qaoa want to call like:
            qaoa.set_symbols
            qaoa.set_beta_values etc...

        CHALLENGE: how to load class functions from sub-module?
        """
        if qalgorithm == "basics":
            self.basics.options = { "append": True,
                                    "start": None,
                                    "end": None,
                                    "n_exact" : [1, 2],
                                    "b_exact" : [0, 0],
                                    "cc_exact": False}
            self.basics.options.update(options)
            self.basics.set_circuit(self)
        elif qalgorithm == "hea":
            self.hea.options = {"append": False,
                                "p": 1,
                                "parametrisation" : 'joint',
                                "variables": {'a', 'x', 'z', 'phi', 'theta'},
                                "1QubitGate": lambda a, x, z: cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a),
                                "2QubitGate": cirq.FSimGate}
            self.hea.options.update(options)
            self.hea.set_symbols(self)
            self.hea.set_circuit(self)
            self.basics.rm_unused_cpv(self)  
            self.basics.add_missing_cpv(self)
        elif qalgorithm == "qaoa":
            # set symbols gets as parameter QAOA repetitions p
            #This needs some further revisions as some parts are not very general yet
            self.qaoa.options = {"append": False,
                                "p": 1,
                                "H_layer": True,
                                "i0": 0}
            self.qaoa.options.update(options)
            self.qaoa.set_symbols(self)
            self.qaoa.set_circuit(self)
        else:
            assert (
                False
            ), "Invalid quantum algorithm, received: '{}', allowed is \n \
                'basics', 'hea', 'qaoa'".format(
                qalgorithm
            )

    def set_circuit_param_values(self, new_values):
        assert np.size(new_values) == np.size(
            self.circuit_param
        ), "np.size(new_values) != np.size(self.circuit_param), {} != {}".format(
            np.size(new_values), np.size(self.circuit_param)
        )
        self.circuit_param_values = new_values

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

    def glue_circuit(self, axis: bool = 0, repetitions: int = 2):
        super().glue_circuit(axis, repetitions)

        #In addition we need to reset j_y_v, j_y_h  h and the hamiltonian
        self.j_y_v=np.tile(self.j_y_v, np.add((1, 1) , (repetitions-1) *(1-axis,axis)))
        self.j_y_h=np.tile(self.j_y_h, np.add((1, 1) , (repetitions-1) *(1-axis,axis)))
        self.j_z_v=np.tile(self.j_z_v, np.add((1, 1) , (repetitions-1) *(1-axis,axis)))
        self.j_z_h=np.tile(self.j_z_h, np.add((1, 1) , (repetitions-1) *(1-axis,axis)))
        self.h =np.tile(self.h, np.add((1, 1) , (repetitions-1) *(1-axis,axis)))
        self._set_hamiltonian()

        # As well as erase eig_val, eig_vec and _Ut as those do not make sense anymore:
        self.eig_val = None
        self.eig_vec = None
        self._Ut = None