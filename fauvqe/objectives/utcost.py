from typing import Literal, Tuple, Dict, Optional
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
                "order"         -> int
                    Trotter approximation order (Exact if 0 or negative)
                "batch_wavefunctions"  -> np.ndarray      if None Use U csot, otherwise batch wavefunctions for random batch cost
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
                    order: int = 0,
                    batch_wavefunctions: Optional[np.ndarray] = None):
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
        # U: Literal["Exact", "Trotter"] = "Exact" is not used yet, this requires implementation of Trotter circuit
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
        self.t = t
        super().__init__(model)
        
        self._order = order
        self._N = 2**np.size(model.qubits)
        self._initials = batch_wavefunctions
        if batch_wavefunctions is None:
            self.batch_size = 0
        else:
            assert(np.size(batch_wavefunctions[0,:]) == 2**np.size(model.qubits)),\
                "Dimension of given batch_wavefunctions do not fit to provided model; n from wf: {}, n qubits: {}".\
                    format(np.log2(np.size(batch_wavefunctions[0,:])), np.size(model.qubits))
            self.batch_size = np.size(batch_wavefunctions[:,0])
            if(self._order != 0):
                self._init_trotter_circuit()
            self._init_batch_wfcts()

    def _init_trotter_circuit(self):
        self.trotter_circuit = cirq.Circuit()
        hamiltonian = self.model.hamiltonian
        print(hamiltonian)
        for pstr in hamiltonian._linear_dict:
            #temp encodes each of the PauliStrings in the PauliSum hamiltonian which need to be turned into gates
            temp = 1
            for pauli in pstr:
                temp = temp * pauli[1](pauli[0])
            self.trotter_circuit.append(temp**np.real(2/np.pi * self.t * hamiltonian._linear_dict[pstr] / self._order))
        self.trotter_circuit = self._order * self.trotter_circuit
    
    def _init_batch_wfcts(self):
        if(self._order < 1):
            self._outputs = (self._Ut @ self._initials.T).T
        else:
            self._outputs = np.zeros(shape=self._initials.shape, dtype=np.complex128)
            #Didn't find any cirq function which accepts a batch of initials
            for k in range(self._initials.shape[0]):
                self._outputs[k] = self.model.simulator.simulate(
                    self.trotter_circuit,
                    initial_state=self._initials[k]
                ).state_vector()

    def evaluate(self, wavefunction: np.ndarray, indices: Optional[list] = None) -> np.float64:
        # Here we already have the correct model._Ut
        if self.batch_size == 0 and indices == None:
            #Calculation via Forbenius norm
            #Then the "wavefunction" is the U(t) via VQE
            return 1 - abs(np.trace(np.matrix.getH(self._Ut) @ wavefunction)) / self._N
        else:
            assert type(indices) is not None, 'Please provide indices for batch'
            print(np.conjugate(wavefunction)*self._outputs[indices])
            return 1/len(indices) * np.sum(1 - abs(np.sum(np.conjugate(wavefunction)*self._outputs[indices], axis=1)))

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
                "batch_wavefunctions": self._initials
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)