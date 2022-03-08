from __future__ import annotations

# External imports
import cirq
from collections.abc import Iterable
import importlib
import itertools
from numbers import Real
import numpy as np
from typing import Tuple, Dict, List

# Internal import
from fauvqe.models.abstractmodel import AbstractModel



class SpinModel(AbstractModel):
    """
    2D SpinModel class inherits AbstractModel
    is mother of different quantum circuit methods
    """
    trotter  = importlib.import_module("fauvqe.models.circuits.trotter")
    basics  = importlib.import_module("fauvqe.models.circuits.basics")
    hea  = importlib.import_module("fauvqe.models.circuits.hea")
    qaoa = importlib.import_module("fauvqe.models.circuits.qaoa")

    def __init__(self, 
                 qubittype, 
                 n,
                 j_v,
                 j_h,
                 h,
                 two_q_gates: List[cirq.PauliSum],
                 one_q_gates: List[cirq.PauliSum],
                 t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j_v vertical j's - same order as two_q_gates
        j_h horizontal j's - same order as two_q_gates
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
        self._set_jh(j_v, j_h, h, two_q_gates, one_q_gates)
        self._set_hamiltonian()
        super().set_simulator()
        self.t = t
        
    def _set_jh(self, j_v, j_h, h, two_q_gates, one_q_gates):
        # convert input to numpy array to be sure
        j_v = np.array(j_v)
        # J vertical needs one row/horizontal line less
        # NEED FOR IMPROVEMENT
        assert (j_v.shape == (len(two_q_gates), *(self.n - np.array((1, 0))) )) or (
            j_v.shape == (len(two_q_gates), *self.n)
        ), "Error in SpinModel._set_jh(): j_v.shape != (len(two_q_gates), n - {{ (1,0), (0,0)}}), {} != {}".format(
            j_v.shape, (len(two_q_gates), *(self.n - np.array((1, 0))))
        )
        self.j_v = np.transpose(j_v, (1, 2, 0))
        
        # convert input to numpy array to be sure
        j_h = np.array(j_h)
        # J horizontal needs one column/vertical line less#
        # NEED FOR IMPROVEMENT
        assert (j_h.shape == (len(two_q_gates), *(self.n - np.array((0, 1))) )) or (
            j_h.shape == (len(two_q_gates), *self.n)
        ), "Error in SpinModel._set_jh(): j_h.shape != (len(two_q_gates), n - {{ (1,0), (0,0)}}), {} != {}".format(
            j_h.shape, (len(two_q_gates), *(self.n - np.array((1, 0))))
        )
        self.j_h = np.transpose(j_h, (1, 2, 0))

        # Set boundaries:
        self.boundaries = np.array((self.n[0] - j_v.shape[1], self.n[1] - j_h.shape[2]))

        # convert input to numpy array to be sure
        h = np.array(h)
        assert (
            h.shape == (len(one_q_gates), *self.n)
        ), "Error in SpinModel._set_jh():: h.shape != (len(one_q_gates), n), {} != {}".format(h.shape, (len(one_q_gates), *self.n))
        self.h = np.transpose(h, (1, 2, 0))

    def _set_hamiltonian(self, reset: bool = True):
        """
            Append or Reset Hamiltonian

            Create a cirq.PauliSum object fitting to j_v, j_h, h  
        """
        if reset:
            self.hamiltonian = cirq.PauliSum()

        #Conversion currently necessary as numpy type * cirq.PauliSum fails
        j_v = self.j_v.tolist()
        j_h = self.j_h.tolist()
        h = self.h.tolist()
        
        # 1. Sum over inner bounds
        for g in range(len(self._two_q_gates)):
            for i in range(self.n[0] - 1):
                for j in range(self.n[1] - 1):
                    #print("i: \t{}, j: \t{}".format(i,j))
                    self.hamiltonian -= j_v[i][j][g]*self._two_q_gates[g](self.qubits[i][j], self.qubits[i+1][j])
                    self.hamiltonian -= j_h[i][j][g]*self._two_q_gates[g](self.qubits[i][j], self.qubits[i][j+1])
        
        for g in range(len(self._two_q_gates)):
            for i in range(self.n[0] - 1):
                j = self.n[1] - 1
                self.hamiltonian -= j_v[i][j][g]*self._two_q_gates[g](self.qubits[i][j], self.qubits[i+1][j])
        
        for g in range(len(self._two_q_gates)):
            for j in range(self.n[1] - 1):
                i = self.n[0] - 1
                self.hamiltonian -= j_h[i][j][g]*self._two_q_gates[g](self.qubits[i][j], self.qubits[i][j+1])
        
        #2. Sum periodic boundaries
        if self.boundaries[1] == 0:
            for g in range(len(self._two_q_gates)):
                for i in range(self.n[0]):
                    j = self.n[1] - 1
                    self.hamiltonian -= j_h[i][j][g]*self._two_q_gates[g](self.qubits[i][j], self.qubits[i][0])
        
        if self.boundaries[0] == 0:
            for g in range(len(self._two_q_gates)):
                for j in range(self.n[1]):
                    i = self.n[0] - 1
                    self.hamiltonian -= j_v[i][j][g]*self._two_q_gates[g](self.qubits[i][j], self.qubits[0][j])
        
        # 3. Add external field
        for g in range(len(self._one_q_gates)):
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    self.hamiltonian -= h[i][j][g]*self._one_q_gates[g](self.qubits[i][j])

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
            # Defaults for 1 and 2 qubit gates
            if len(self._one_q_gates) < 2:
                _1Qvariables = [['a' , 'x', 'z']]
            else:
                _1Qvariables = [['a' + str(g) + '_', 'x'+ str(g) + '_', 'z' + str(g) + '_'] for g in range(len(self._one_q_gates))]

            if len(self._two_q_gates) < 2:
                _2Qvariables = [['phi', 'theta']]
            else:
                _2Qvariables = [['phi' + str(g) + '_', 'theta' + str(g) + '_'] for g in range(len(self._two_q_gates))]


            self.hea.options = {"append": False,
                                "p": 1,
                                "parametrisation" : 'joint',
                                "fully_connected" : False,
                                "1Qvariables": _1Qvariables,
                                "2Qvariables": _2Qvariables,
                                "1QubitGates": [lambda a, x, z: cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a) for g in range(len(self._one_q_gates))],
                                "2QubitGates" : [lambda phi, theta: cirq.FSimGate(phi=phi, theta=theta)  for g in range(len(self._two_q_gates))],
                               }
            
            # Convert options input to correct format
            for key, nested_level in [  ["1Qvariables", 2], 
                                        ["2Qvariables", 2],
                                        ["1QubitGates", 1],
                                        ["2QubitGates", 1]]:
                options = self._update2nestedlist(options, key, nested_level)

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
        elif qalgorithm == "qaoa":
            # set symbols gets as parameter QAOA repetitions p
            #This needs some further revisions as some parts are not very general yet
            self.qaoa.options = {"append": False,
                                "p": 1,
                                "H_layer": True,
                                "fully_connected" : False,
                                "i0": 0,
                                "1QubitGates": [ lambda q, theta: cirq.Z(q)**(theta)],
                                "2QubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta)]
                                }
            self.qaoa.options.update(options)
            self.qaoa.set_symbols(self)
            self.qaoa.set_circuit(self)
        else:
            assert (
                False
            ), "Invalid quantum algorithm, received: '{}', allowed is \n \
                'basics', 'hea', 'qaoa', 'trotter'".format(
                qalgorithm
            )

    def set_circuit_param_values(self, new_values):
        assert np.size(new_values) == np.size(
            self.circuit_param
        ), "np.size(new_values) != np.size(self.circuit_param), {} != {}".format(
            np.size(new_values), np.size(self.circuit_param)
        )
        self.circuit_param_values = new_values
    
    def glue_circuit(self, axis: bool = 0, repetitions: int = 2):
        super().glue_circuit(axis, repetitions)

        #In addition we need to reset j_v, j_h  h and the hamiltonian
        self.j_v=np.transpose(np.tile(np.transpose(self.j_v, (2,0,1)), np.add((1, 1) , (repetitions-1) *(1-axis,axis))), (1,2,0))
        self.j_h=np.transpose(np.tile(np.transpose(self.j_h, (2,0,1)), np.add((1, 1) , (repetitions-1) *(1-axis,axis))), (1,2,0))
        self.h = np.transpose(np.tile(np.transpose(self.h, (2,0,1)), np.add((1, 1) , (repetitions-1) *(1-axis,axis))), (1,2,0))
        self._set_hamiltonian()

        # As well as erase eig_val, eig_vec and _Ut as those do not make sense anymore:
        self.eig_val = None
        self.eig_vec = None
        self._Ut = None
    
    def energy_2q(self, j_v, j_h) -> np.ndarray:
        n_sites = self.n[0] * self.n[1]
        Z = np.array([(-1) ** (np.arange(2 ** n_sites) >> i) for i in range(n_sites - 1, -1, -1)])
        
        ZZ_filter = np.zeros(
            2 ** (n_sites), dtype=np.float64
        )
        
        # 1. Sum over inner bounds
        for i in range(self.n[0] - 1):
            for j in range(self.n[1] - 1):
                ZZ_filter += j_v[i, j] * Z[i * self.n[1] + j] * Z[(i + 1) * self.n[1] + j]
                ZZ_filter += j_h[i, j] * Z[i * self.n[1] + j] * Z[i * self.n[1] + (j + 1)]

        for i in range(self.n[0] - 1):
            j = self.n[1] - 1
            ZZ_filter += j_v[i, j] * Z[i * self.n[1] + j] * Z[(i + 1) * self.n[1] + j]

        for j in range(self.n[1] - 1):
            i = self.n[0] - 1
            ZZ_filter += j_h[i, j] * Z[i * self.n[1] + j] * Z[i * self.n[1] + (j + 1)]

        # 2. Sum periodic boundaries
        if self.boundaries[1] == 0:
            for i in range(self.n[0]):
                j = self.n[1] - 1
                ZZ_filter += j_h[i, j] * Z[i * self.n[1] + j] * Z[i * self.n[1]]

        if self.boundaries[0] == 0:
            for j in range(self.n[1]):
                i = self.n[0] - 1
                ZZ_filter += j_v[i, j] * Z[i * self.n[1] + j] * Z[j]
        
        return ZZ_filter
    
    def energy_1q(self, h) -> np.ndarray:
        n_sites = self.n[0] * self.n[1]
        Z = np.array([(-1) ** (np.arange(2 ** n_sites) >> i) for i in range(n_sites - 1, -1, -1)])
        
        return h.reshape(n_sites).dot(Z)
    
    def energy(self, j_v, j_h, h) -> np.ndarray:
        return self.energy_1q(h) + self.energy_2q(j_v, j_h)
    
    def normalise(self, spread: float = 2) -> None:
        '''
        Scales and shifts the system Hamiltonian, B, J and shift to achieve
        specified minimum and maximum energies in the Hamiltonian
        eigenspectrum.
        
        Overrides AbstractModel's normalise() function
        '''
        _n = np.size(self.qubits)
        _N = 2**_n
        if np.size(self.eig_val) != _N or \
        (np.shape(self.eig_vec) != np.array((_N, _N)) ).all():
            self.diagonalise(solver = "numpy")
        previous_spread = (self.eig_val[-1] - self.eig_val[0])
        scale = spread / previous_spread
        super().normalise(spread)
        self.j_v *= scale
        self.j_h *= scale
        self.h *= scale
    
    def copy(self) -> SpinModel:
        self_copy = SpinModel( self.qubittype,
                self.n,
                np.transpose(self.j_v, (2, 0, 1)),
                np.transpose(self.j_h, (2, 0, 1)),
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
                "j_v": np.transpose(self.j_v, (2, 0, 1)),
                "j_h": np.transpose(self.j_h, (2, 0, 1)),
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

    def _update2nestedlist(self, options: dict(), key, new_nested_level: int = 1):
        if options.get(key) is not None:
            if isinstance(options.get(key), Iterable):
                _tmp = list(options.get(key))
                Is_nested_level = self._nest_level(_tmp) - 1
            else:
                _tmp = [options.get(key) ]
                Is_nested_level=0

            for i in range(1, new_nested_level-Is_nested_level):
                _tmp = [_tmp]
            options.update({key: _tmp})

        return options

    def _nest_level(self, lst):
        if not isinstance(lst, list):
            return 0
        if not lst:
            return 1
        return max(self._nest_level(item) for item in lst) + 1
