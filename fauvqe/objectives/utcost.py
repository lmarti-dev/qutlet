"""
    Implementation of the Frobenius distance between a given approximate time evolution and the exact time evolution of the system hamiltonian as objective function for an AbstractModel object.
"""
from typing import Literal, Dict, Optional, List
from numbers import Integral, Real

from joblib import delayed, Parallel
from multiprocessing import cpu_count

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
                "tnumber"         -> np.uint
                    Trotter number (Exact if 0 or negative)

                "initial_wavefunctions"  -> np.ndarray      if None Use exact UtCost, otherwise batch wavefunctions for random batch cost. Also overwrites evaluate(), the NotImplementedError should never be raised.
                
                "time_steps"  -> List[Int]      List of numbers of circuit runs to be considered in UtCost in ascending order
                
                "use_progress_bar"  -> bool      Determines whether to use a progress bar while initialising the batch wavefunctions (python module tqdm required)

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
                    tnumber: np.uint = 0,
                    initial_wavefunctions: Optional[np.ndarray] = None,
                    time_steps: List[int] = [1],
                    use_progress_bar: bool = False,
                    dtype: np.dtype = np.complex128):
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
        self._tnumber = tnumber
        self._use_progress_bar = use_progress_bar
        self._N = 2**np.size(model.qubits)
        self._time_steps = time_steps
        self._dtype = dtype
        
        if self._tnumber == 0:
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
            self.trotter_circuit.append(temp**np.real(2/np.pi * self.t * hamiltonian._linear_dict[pstr] / self._tnumber))
        #Copy the Trotter layer *tnumber times.
        #self.trotter_circuit = qsimcirq.QSimCircuit(self._tnumber * self.trotter_circuit)
        self.trotter_circuit = self._tnumber * self.trotter_circuit
    
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
        self._output_wavefunctions = np.empty(shape=( len(self._time_steps), *self._initial_wavefunctions.shape), dtype=self._dtype)
        if(self._tnumber < 1):
            for step in range(len(self._time_steps)):
                self._output_wavefunctions[step] = (np.linalg.matrix_power(self._Ut, self._time_steps[step]) @ self._initial_wavefunctions.T).T
        else:
            pbar = self.create_range(self.batch_size, self._use_progress_bar)
            #Didn't find any cirq function which accepts a batch of initials
            for step in range(len(self._time_steps)):
                if(step != 0):
                    ini = self._output_wavefunctions[step - 1]
                    mul = self._time_steps[step] - self._time_steps[step - 1]
                else:
                    ini = self._initial_wavefunctions
                    mul = self._time_steps[step]
                #Use multiprocessing
                tmp = Parallel(n_jobs=min(cpu_count(), self.batch_size))(
                delayed(self.model.simulator.simulate)( mul * self.trotter_circuit, initial_state=ini[k]) for k in pbar
                )
                for k in range(self._initial_wavefunctions.shape[0]):
                    self._output_wavefunctions[step][k] = tmp[k].state_vector() / np.linalg.norm(tmp[k].state_vector())
            if(self._use_progress_bar):
                pbar.close()

    def evaluate(self):
        raise NotImplementedError
            
    def evaluate_op(self, wavefunction: np.ndarray) -> np.float64:
        cost = 0
        for step in range(len(self._time_steps)):
            cost = cost + 1 - abs(np.trace( np.linalg.matrix_power( np.matrix.getH(self._Ut), self._time_steps[step]) @ np.linalg.matrix_power(wavefunction, self._time_steps[step]))) / self._N
        return 1 / len(self._time_steps) * cost
    
    def evaluate_batch(self, wavefunction: np.ndarray, options: dict = {'indices': None}) -> np.float64:
        #assert (options['indices'] is not None), 'Please provide indices for batch'
        #assert wavefunction.size == (len(self._time_steps) * len(options['indices']) * self._N), 'Please provide one wavefunction for each time step. Expected: {} Received: {}'.\
        #    format( len(self._time_steps), wavefunction.size / self._N )
        cost = 0
        for step in range(len(self._time_steps)):
            cost += 1/len(options['indices']) * np.sum(1 - abs(np.sum(np.conjugate(wavefunction[step])*self._output_wavefunctions[step][options['indices']], axis=1)))
        return 1 / len(self._time_steps) * cost

    #Need to overwrite simulate from parent class in order to work
    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        #return unitary if self.batch_size == 0
        if self.batch_size == 0:
            return cirq.resolve_parameters(self._model.circuit, param_resolver).unitary()
        else:
            output_state = np.empty(shape=(len(self._time_steps), *initial_state.shape), dtype = self._dtype)
            ini = initial_state
            mul = self._time_steps[0]
            wf = self._model.simulator.simulate(
                    mul * self._model.circuit,
                    param_resolver=param_resolver,
                    initial_state=ini,
                    ).state_vector()
            output_state[0] = wf/np.linalg.norm(wf)
            for step in range(1, len(self._time_steps)):
                ini = output_state[step-1]
                mul = self._time_steps[step] - self._time_steps[step-1]
                wf = self._model.simulator.simulate(
                    mul * self._model.circuit,
                    param_resolver=param_resolver,
                    initial_state=ini,
                    ).state_vector()
                output_state[step] = wf/np.linalg.norm(wf)
            return output_state
            
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "t": self.t, 
                "tnumber": self._tnumber,
                "initial_wavefunctions": self._initial_wavefunctions
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)