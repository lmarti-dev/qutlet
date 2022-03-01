from __future__ import annotations

import importlib
from typing import Tuple, Dict, List
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.abstractmodel import AbstractModel



class SpinModelFC(AbstractModel):
    """
    2D SpinModel class for fully connected interaction graphs inherits AbstractModel
    is mother of different quantum circuit methods
    """
    basics  = importlib.import_module("fauvqe.models.circuits.basics")
    trotter  = importlib.import_module("fauvqe.models.circuits.trotter")
    cooling  = importlib.import_module("fauvqe.models.circuits.cooling")
    hea  = importlib.import_module("fauvqe.models.circuits.hea")
    qaoa = importlib.import_module("fauvqe.models.circuits.qaoa")

    def __init__(self, 
                 qubittype, 
                 n,
                 j,
                 h,
                 two_q_gates: List[cirq.PauliSum],
                 one_q_gates: List[cirq.PauliSum],
                 t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j full connected interaction matrix j's - same order as two_q_gates
        h  strength external fields - same order as one_q_gates
        two_q_gates: list of 2 Qubit Gates
        one_q_gates: list of single Qubit Gates
        t: Simulation Time
        """
        # convert all input to np array to be sure
        super().__init__(qubittype, np.array(n))
        self.circuit_param = None
        self.circuit_param_values = np.array([])
        self._two_q_gates = two_q_gates
        self._one_q_gates = one_q_gates
        self._set_jh(j, h, two_q_gates, one_q_gates)
        self._set_hamiltonian()
        super().set_simulator()
        self.t = t

    def _check_symmetric(self, j):
        return np.allclose(
            j, 
            np.transpose(j, (0, 3, 4, 1, 2)),
            atol=1e-13
        )
    
    def _set_jh(self, j, h, two_q_gates, one_q_gates):
        # convert input to numpy array to be sure
        j = np.array(j)
        assert (
            j.shape == (len(two_q_gates), *self.n, *self.n)
        ), "Error in SpinModel._set_jh(): j.shape != (len(two_q_gates), n, n ), {} != {}".format(
            j.shape, (len(two_q_gates), *(self.n), *(self.n))
        )
        assert self._check_symmetric(j), "Interaction graph is not symmetric: " + str(j)
        self.j = np.transpose(j, (1, 2, 3, 4, 0))
        
        # convert input to numpy array to be sure
        h = np.array(h)
        assert (
            h.shape == (len(one_q_gates), *self.n)
        ), "Error in SpinModel._set_jh():: h.shape != (len(one_q_gates), n), {} != {}".format(h.shape, (len(one_q_gates), *self.n))
        self.h = np.transpose(h, (1, 2, 0))

    def _set_hamiltonian(self, reset: bool = True):
        """
            Append or Reset Hamiltonian

            Create a cirq.PauliSum object fitting to j, h  
        """
        if reset:
            self.hamiltonian = cirq.PauliSum()

        #Conversion currently necessary as numpy type * cirq.PauliSum fails
        js = self.j.tolist()
        hs = self.h.tolist()
        
        # 1. Sum over 2QGates
        for g in range(len(self._two_q_gates)):
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    #k==i, l>j
                    for l in range(j+1, self.n[1], 1):
                        if(js[i][j][i][l][g] != 0):
                            self.hamiltonian -= js[i][j][i][l][g] * self._two_q_gates[g](self.qubits[i][j], self.qubits[i][l])
                    #k>i
                    for k in range(i+1, self.n[0], 1):
                        for l in range(self.n[1]):
                            if(js[i][j][k][l][g] != 0):
                                self.hamiltonian -= js[i][j][k][l][g] * self._two_q_gates[g](self.qubits[i][j], self.qubits[k][l])
                    
        # 2. Add external field
        for g in range(len(self._one_q_gates)):
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    if(hs[i][j][g]!=0):
                        self.hamiltonian -= hs[i][j][g]*self._one_q_gates[g](self.qubits[i][j])
    
    def set_circuit(self, qalgorithm, options: dict = {}):
        if qalgorithm == "hea":
            self.hea.options = {"append": False,
                                "p": 1,
                                "parametrisation" : 'joint',
                                "fully_connected" : True,
                                "1Qvariables": [['a' + str(g) + '_', 'x'+ str(g) + '_', 'z' + str(g) + '_'] for g in range(len(self._one_q_gates))],
                                "2Qvariables": [['phi' + str(g) + '_', 'theta' + str(g) + '_'] for g in range(len(self._two_q_gates))],
                                "1QubitGates": [lambda a, x, z: cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a) for g in range(len(self._one_q_gates))],
                                "2QubitGates" : [cirq.FSimGate for g in range(len(self._two_q_gates))],
                               }
            self.hea.options.update(options)
            self.hea.set_symbols(self)
            self.hea.set_circuit(self)
            self.basics.rm_unused_cpv(self)  
            self.basics.add_missing_cpv(self)
        elif qalgorithm == "trotter":
            self.trotter.options = { "append": False,
                                    "q":1,
                                    "m":1
                                  }
            self.trotter.options.update(options)
            self.trotter.set_circuit(self)
        elif qalgorithm == "cooling":
            self.cooling.options = { "append": False,
                                    "K":1,
                                    "emin":None,
                                    "emax":None,
                                    "m":None,
                                    "time_steps":1,
                                  }
            self.cooling.options.update(options)
            self.cooling.set_circuit(self)
        elif qalgorithm == "qaoa":
            # set symbols gets as parameter QAOA repetitions p
            #This needs some further revisions as some parts are not very general yet
            self.qaoa.options = {"append": False,
                                "p": 1,
                                "H_layer": True,
                                "i0": 0,
                                "fully_connected" : True,
                                "1QubitGates": [ lambda q, theta: cirq.Z(q)**(theta)],
                                "2QubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta)],
                                }
            self.qaoa.options.update(options)
            self.qaoa.set_symbols(self)
            self.qaoa.set_circuit(self)
        else:
            assert (
                False
            ), "Invalid quantum algorithm, received: '{}', allowed is \n \
                'basics', 'hea', 'qaoa', 'trotter', 'cooling'".format(
                qalgorithm
            )

    def set_circuit_param_values(self, new_values):
        assert np.size(new_values) == np.size(
            self.circuit_param
        ), "np.size(new_values) != np.size(self.circuit_param), {} != {}".format(
            np.size(new_values), np.size(self.circuit_param)
        )
        self.circuit_param_values = new_values
    
    def energy_2q(self, js) -> np.ndarray:
        n_sites = self.n[0] * self.n[1]
        Z = np.array([(-1) ** (np.arange(2 ** n_sites) >> i) for i in range(n_sites - 1, -1, -1)])
        
        ZZ_filter = np.zeros(
            2 ** (n_sites), dtype=np.float64
        )
        
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                #k==i, l>j
                for l in range(j+1, self.n[1], 1):
                    ZZ_filter += js[i, j, i, l] * Z[i * self.n[1] + j] * Z[i * self.n[1] + l]
                #k>i
                for k in range(i+1, self.n[0], 1):
                    for l in range(self.n[1]):
                        if ( (i<k) or (i==k and j<=l) ):
                            ZZ_filter += js[i, j, k, l] * Z[i * self.n[1] + j] * Z[k * self.n[1] + l]

        return ZZ_filter
    
    def energy_1q(self, h) -> np.ndarray:
        n_sites = self.n[0] * self.n[1]
        Z = np.array([(-1) ** (np.arange(2 ** n_sites) >> i) for i in range(n_sites - 1, -1, -1)])
        
        return h.reshape(n_sites).dot(Z)
    
    def energy(self, j, h) -> np.ndarray:
        return self.energy_1q(h) + self.energy_2q(j)
    
    def copy(self) -> SpinModelFC:
        self_copy = SpinModelFC( self.qubittype,
                self.n,
                np.transpose(self.j, (4, 0, 1, 2, 3)),
                np.transpose(self.h, (2, 0, 1)),
                self._two_q_gates,
                self._one_q_gates,
                self.t )

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
                "qubittype": self.qubittype,
                "n": self.n,
                "j": np.transpose(self.j, (4, 0, 1, 2, 3)),
                "h": np.transpose(self.h, (2, 0, 1)),
                "two_q_gates": self._two_q_gates,
                "one_q_gates": self._one_q_gates,
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