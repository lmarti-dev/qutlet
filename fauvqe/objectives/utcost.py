"""
    Implementation of the Frobenius distance between a given approximate time evolution and the exact time evolution of the system hamiltonian as objective function for an AbstractModel object.
"""
from typing import Literal, Dict, Optional, List
from numbers import Integral, Real

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
#from fauvqe import Objective, AbstractModel
import cirq

class UtCost(Objective):
    """
    U(t) cost objective

    This class implements as objective the difference between exact U(t) and VQE U(t)
    of the linked model.

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "t"         -> Float
                    t in U(t)
                "order"         -> np.uint
                    Trotter approximation order (Exact if 0 or negative)
                "initial_wavefunctions"  -> np.ndarray      if None Use U csot, otherwise batch wavefunctions for random batch cost
                "sample_size"  -> Int      < 0 -> state vector, > 0 -> number of samples

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <UtCost field=self.field>
    """
    def __init__(   self,
                    model: AbstractModel, 
                    t: Real, 
                    order: np.uint = 0,
                    initial_wavefunctions: Optional[np.ndarray] = None,
                    use_progress_bar: bool = False):
        #Idea of using variable "method" her instead of boolean is that 
        #This allows for more than 2 Calculation methods like:
        #   Cost via exact unitary
        #   Cost via Trotter unitary
        #       -Cost via random batch sampling of these, but exact state vector
        #       -Cost via random batch sampling of these & sampling state
        #
        #   Based on the method, different further parameters are needed hence rather use dict
        # To be implemented: U exact unitatry cost, U exact random batch sampling cost with wf 
        
        #Make sure correct Ut is used
        self.t = t
        super().__init__(model)
        
        self._initial_wavefunctions = initial_wavefunctions
        self._order = order
        self._use_progress_bar = use_progress_bar
        self._N = 2**np.size(model.qubits)
        
        if self._order == 0:
            if t != model.t:
                model.t = t
                model.set_Ut()
                self._Ut = model._Ut.view()
            else:
                try:
                    #Fails if does not exist
                    self._Ut = model._Ut.view()
                except:
                    model.set_Ut()
                    self._Ut = model._Ut.view()
        else:
            assert initial_wavefunctions is not None, 'Please provide batch wavefunctions for Trotter Approximation'
            self._init_trotter_circuit()
        if (initial_wavefunctions is None):
            self.batch_size = 0
            self.evaluate = self.evaluate_op
        else:
            assert(np.size(initial_wavefunctions[0,:]) == 2**np.size(model.qubits)),\
                "Dimension of given batch_wavefunctions do not fit to provided model; n from wf: {}, n qubits: {}".\
                    format(np.log2(np.size(initial_wavefunctions[0,:])), np.size(model.qubits))
            self.batch_size = np.size(initial_wavefunctions[:,0])
            self._init_batch_wfcts()
            self.evaluate = self.evaluate_batch

    def _init_trotter_circuit(self):
        """
        This function initialises the circuit for Trotter approximation and sets self.trotter_circuit
        
        Parameters
        ----------
        self
        
        Returns
        ---------
        void
        """
        self.trotter_circuit = cirq.Circuit()
        hamiltonian = self.model.hamiltonian
        #Loop through all the addends in the PauliSum Hamiltonian
        for pstr in hamiltonian._linear_dict:
            #temp encodes each of the PauliStrings in the PauliSum hamiltonian which need to be turned into gates
            temp = 1
            #Loop through Paulis in the PauliString (pauli[1] encodes the cirq gate and pauli[0] encodes the qubit on which the gate acts)
            for pauli in pstr:
                temp = temp * pauli[1](pauli[0])
            #Append the PauliString gate in temp to the power of the time step * coefficient of said PauliString. The coefficient needs to be multiplied by a correction factor of 2/pi in order for the PowerGate to represent a Pauli exponential.
            self.trotter_circuit.append(temp**np.real(2/np.pi * self.t * hamiltonian._linear_dict[pstr] / self._order))
        #Copy the Trotter layer *order times.
        #self.trotter_circuit = qsimcirq.QSimCircuit(self._order * self.trotter_circuit)
        self.trotter_circuit = self._order * self.trotter_circuit
    
    def _init_batch_wfcts(self):
        """
        This function initialises the initial and output batch wavefunctions as sampling data and sets self._output_wavefunctions.
        
        Parameters
        ----------
        self
        
        Returns
        ---------
        void
        """
        if(self._order < 1):
            self._output_wavefunctions = (self._Ut @ self._initial_wavefunctions.T).T
        else:
            pbar = self.create_range(self.batch_size, self._use_progress_bar)
            self._output_wavefunctions = np.zeros(shape=self._initial_wavefunctions.shape, dtype=np.complex128)
            #Didn't find any cirq function which accepts a batch of initials
            for k in pbar:
                self._output_wavefunctions[k] = self.model.simulator.simulate(
                    self.trotter_circuit,
                    initial_state=self._initial_wavefunctions[k]
                    #dtype=np.complex128
                ).state_vector()
            if(self._use_progress_bar):
                pbar.close()
            #self.trotter_circuit = qsimcirq.QSimCircuit(self.trotter_circuit)
            #start = time()
            #for k in range(self._initial_wavefunctions.shape[0]):
            #    self._output_wavefunctions[k] = self.trotter_circuit.final_state_vector(
            #        initial_state=self._initial_wavefunctions[k],
            #        dtype=np.complex64
            #    )
            #end = time()
            #print(end-start)
    
    def evaluate(self, wavefunction: np.ndarray, options: dict = {'indices': None}) -> np.float64:
        return self.evaluate_op(wavefunction, options)
    
    def evaluate_op(self, wavefunction: np.ndarray) -> np.float64:
        return 1 - abs(np.trace(np.matrix.getH(self._Ut) @ wavefunction)) / self._N
    
    def evaluate_batch(self, wavefunction: np.ndarray, options: dict = {'indices': None}) -> np.float64:
        return 1/len(options['indices']) * np.sum(1 - abs(np.sum(np.conjugate(wavefunction)*self._output_wavefunctions[options['indices']], axis=1)))

    #Need to overwrite simulate from parent class in order to work
    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        #return unitary if self.batch_size == 0
        if self.batch_size == 0:
            return cirq.resolve_parameters(self._model.circuit, param_resolver).unitary()
        else:
            return super().simulate(param_resolver, initial_state)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "t": self.t, 
                "order": self._order,
                "initial_wavefunctions": self._initial_wavefunctions
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)