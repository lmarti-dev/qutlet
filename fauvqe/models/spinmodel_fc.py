from __future__ import annotations

import importlib
from typing import Tuple, Dict, List, TYPE_CHECKING
from numbers import Real
import itertools

import numpy as np
import cirq

from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.models.spinmodel import SpinModel

if TYPE_CHECKING:
    from fauvqe.models.coolingmodel import CoolingModel

class SpinModelFC(AbstractModel):
    """
    2D SpinModel class for fully connected interaction graphs inherits AbstractModel
    is mother of different quantum circuit methods
    
    Parameters
    ----------
    qubittype: str, indicates QubitType supported by cirq library
    n: List[int], system size (row and column)
    j: np.array, interaction constants for TwoQubitGates
    h: np.array, field strengths for SingleQubitGates
    TwoQubitGates: List[cirq.PauliSum], List of entangling cirq gates
    SingleQubitGates: List[cirq.PauliSum], List of single qubit cirq gates
    t: Real = 0, Simulation time
    
    Methods
    ----------
    _check_symmetric(self, j: np.array)
        Returns
        ---------
        bool:
            Checks whether interaction constant adjacency matric is symmetric
    
    _combine_jh(self): List[np.array]
        Returns
        ---------
        List[np.array]:
            Arrays defining interaction js and field strengths h
    
    _set_jh(self, j, h): void
        Returns
        ---------
        void
    
    _set_hamiltonian(self, reset: bool): void
        Returns
        ---------
        void
    
    set_circuit(self, qalgorithm: str, options: dict): void
        Returns
        ---------
        void
    """
    basics  = importlib.import_module("fauvqe.circuits.basics")
    cooling  = importlib.import_module("fauvqe.circuits.cooling")
    hea  = importlib.import_module("fauvqe.circuits.hea")
    qaoa = importlib.import_module("fauvqe.circuits.qaoa")
    trotter  = importlib.import_module("fauvqe.circuits.trotter")

    def __init__(self, 
                 qubittype: str, 
                 n: List[int],
                 j: np.array,
                 h: np.array,
                 TwoQubitGates: List[cirq.PauliSum],
                 SingleQubitGates: List[cirq.PauliSum],
                 t: Real = 0):
        """
        qubittype as defined in AbstractModel
        n number of qubits
        j full connected interaction matrix j's - same order as TwoQubitGates
        h  strength external fields - same order as SingleQubitGates
        TwoQubitGates: list of 2 Qubit Gates
        _SingleQubitGates: list of single Qubit Gates
        t: Simulation Time
        """
        # convert all input to np array to be sure
        super().__init__(qubittype, np.array(n))
        self.circuit_param = None
        self.circuit_param_values = np.array([])
        self._TwoQubitGates = TwoQubitGates
        self._SingleQubitGates = SingleQubitGates
        self._set_jh(j, h)
        self.t = t
        self._set_hamiltonian()
        super().set_simulator()

    def _check_symmetric(self, j: np.array) -> bool:
        """
        Check whether handed interaction adjacency matrix is symmetric and returns result as boolean variable

        Parameters
        ----------
        self
        j: np.array, interaction adjacency matrix
        
        Returns
        -------
        j == j.T: bool
        """
        return np.allclose(
            j, 
            np.transpose(j, (0, 3, 4, 1, 2)),
            atol=1e-13
        )
    
    def _set_jh(self, j: np.array, h: np.array) -> None:
        """
        Set interaction and field strengths after checking shape asserts

        Parameters
        ----------
        self
        j: np.array, interaction adjacency matrix
        h: field strength tensor
        
        Returns
        -------
        void
        """
        #TODO Allow here also for a list of qubits and interaction strength -> avoid large 0 matrices and also easier for sparse diagonalisation
        _TwoQubitGates = self._TwoQubitGates
        _SingleQubitGates = self._SingleQubitGates
        # convert input to numpy array to be sure
        j = np.array(j)
        assert (
            j.shape == (len(_TwoQubitGates), *self.n, *self.n)
        ), "Error in SpinModel._set_jh(): j.shape != (len(_TwoQubitGates), n, n ), {} != {}".format(
            j.shape, (len(_TwoQubitGates), *(self.n), *(self.n))
        )
        assert self._check_symmetric(j), "Interaction graph is not symmetric: " + str(j)
        self.j = np.transpose(j, (1, 2, 3, 4, 0))
        
        # convert input to numpy array to be sure
        h = np.array(h)
        assert (
            h.shape == (len(_SingleQubitGates), *self.n)
        ), "Error in SpinModel._set_jh():: h.shape != (len(_SingleQubitGates), n), {} != {}".format(h.shape, (len(_SingleQubitGates), *self.n))
        self.h = np.transpose(h, (1, 2, 0))

    def _set_hamiltonian(self, reset: bool = True) -> None:
        """
        Append or Reset Hamiltonian; Create a cirq.PauliSum object fitting to j, h  
        
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

        #Conversion currently necessary as numpy type * cirq.PauliSum fails
        js = self.j.tolist()
        hs = self.h.tolist()
        
        # 1. Sum over 2QGates
        for g in range(len(self._TwoQubitGates)):
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    #k==i, l>j
                    for l in range(j+1, self.n[1], 1):
                        if(js[i][j][i][l][g] != 0):
                            self.hamiltonian -= js[i][j][i][l][g] * self._TwoQubitGates[g](self.qubits[i][j], self.qubits[i][l])
                    #k>i
                    for k in range(i+1, self.n[0], 1):
                        for l in range(self.n[1]):
                            if(js[i][j][k][l][g] != 0):
                                self.hamiltonian -= js[i][j][k][l][g] * self._TwoQubitGates[g](self.qubits[i][j], self.qubits[k][l])
                    
        # 2. Add external field
        for g in range(len(self._SingleQubitGates)):
            for i in range(self.n[0]):
                for j in range(self.n[1]):
                    if(hs[i][j][g]!=0):
                        self.hamiltonian -= hs[i][j][g]*self._SingleQubitGates[g](self.qubits[i][j])
    
    def set_circuit(self, qalgorithm: str, options: dict = {}) -> None:
        """
        Append or Reset circuit. Possible circuit types are
            hea: 
            trotter: 
            cooling: 
            qaoa: 
        
        Parameters
        ----------
        self
        qalgorithm: str, determines the circuit type
        options: options handed over to the circuit generating script
        
        Returns
        -------
        void 
        """
        if qalgorithm == "hea":
            self.hea.options = {"append": False,
                                "p": 1,
                                "parametrisation" : 'joint',
                                "is_fully_connected" : True,
                                "SingleQubitVariables": [['a' + str(g) + '_', 'x'+ str(g) + '_', 'z' + str(g) + '_'] for g in range(len(self._SingleQubitGates))],
                                "TwoQubitVariables": [['phi' + str(g) + '_', 'theta' + str(g) + '_'] for g in range(len(self._TwoQubitGates))],
                                "SingleQubitGates": [lambda a, x, z: cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a) for g in range(len(self._SingleQubitGates))],
                                "TwoQubitGates" : [cirq.FSimGate for g in range(len(self._TwoQubitGates))],
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
            assert isinstance(self, CoolingModel)
            self.cooling.options = { "append": False,
                                    "K":1,
                                    "emin":None,
                                    "emax":None,
                                    "m":None,
                                    "q":1,
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
                                "is_fully_connected" : True,
                                "SingleQubitGates": [ lambda q, theta: cirq.Z(q)**(theta)],
                                "TwoQubitGates" : [lambda q1, q2, theta: cirq.ZZ(q1, q2)**(theta)],
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
    
    def set_spectrum_scale(self, spread: float = 2) -> None:
        '''
        Scales and shifts the system Hamiltonian, B, J and shift to achieve
        specified minimum and maximum energies in the Hamiltonian
        eigenspectrum.
        
        Overrides AbstractModel's set_spectrum_scale() function
        '''
        _n = np.size(self.qubits)
        _N = 2**_n
        if np.size(self.eig_val) != _N or \
        (np.shape(self.eig_vec) != np.array((_N, _N)) ).all():
            self.diagonalise(solver = "numpy")
        previous_spread = (self.eig_val[-1] - self.eig_val[0])
        scale = spread / previous_spread
        super().set_spectrum_scale(spread)
        self.j *= scale
        self.h *= scale
    
    def copy(self) -> SpinModelFC:
        self_copy = SpinModelFC( self.qubittype,
                self.n,
                np.transpose(self.j, (4, 0, 1, 2, 3)),
                np.transpose(self.h, (2, 0, 1)),
                self._TwoQubitGates,
                self._SingleQubitGates,
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
                "TwoQubitGates": self._TwoQubitGates,
                "SingleQubitGates": self._SingleQubitGates,
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
    
    @classmethod
    def toFC(cls, model: SpinModel) -> np.ndarray:
        j_h = np.transpose(model.j_h, (2, 0, 1))
        j_v = np.transpose(model.j_v, (2, 0, 1))
        j = np.zeros((model.j_h.shape[-1], *model.n, *model.n))
        
        for g in range(len(j_h)):
            for n0 in range(model.n[0]-1):
                for n1 in range(model.n[1]-1):
                    j[g][n0][n1][n0][(n1+1)] = j_h[g][n0][n1]
                    j[g][n0][(n1+1)][n0][n1] = j_h[g][n0][n1]
                    j[g][n0][n1][(n0+1)][n1] = j_v[g][n0][n1]
                    j[g][(n0+1)][n1][n0][n1] = j_v[g][n0][n1]
            
            for n0 in range(model.n[0]-1):
                n1 = model.n[1]-1
                j[g][n0][n1][(n0+1)][n1] = j_v[g][n0][n1]
                j[g][(n0+1)][n1][n0][n1] = j_v[g][n0][n1]
            
            for n1 in range(model.n[1]-1):
                n0 = model.n[0]-1
                j[g][n0][n1][n0][n1+1] = j_h[g][n0][n1]
                j[g][n0][n1+1][n0][n1] = j_h[g][n0][n1]
            
            if model.boundaries[1] == 0:
                for n0 in range(model.n[0]):
                    n1 = model.n[1]-1
                    j[g][n0][n1][n0][0] = j_h[g][n0][n1]
                    j[g][n0][0][n0][n1] = j_h[g][n0][n1]
            
            if model.boundaries[0] == 0:
                for n1 in range(model.n[1]):
                    n0 = model.n[0]-1
                    j[g][n0][n1][0][n1] = j_v[g][n0][n1]
                    j[g][0][n1][n0][n1] = j_v[g][n0][n1]
        
        '''
        h = np.transpose(model.h, (2, 0, 1))
        
        modelFC = SpinModelFC(
            model.qubittype, 
            model.n,
            j,
            h,
            model._TwoQubitGates,
            model._SingleQubitGates,
            model.t
        )
        
        modelFC.energy_fields = model.energy_fields
        return modelFC
        '''
        return np.transpose(j, (1, 2, 3, 4, 0))