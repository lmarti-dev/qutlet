"""
    Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional
from numbers import Integral, Real

import numpy as np

from fauvqe import Objective, AbstractModel
import cirq

class UtCost(Objective):
    """
    U(t) cost objective

    This class implements as objective the difference between exact U(t) and VQE U(t)
    of the linked model.

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "t"         -> Float    t in U(t)
                "U"         -> Literal  exact or Trotter (only matters if not exists or t wrong)
                "benchm"         -> AbstractModel   Only needed for approximate randomized benchmarking. Defines the benchmark model
                "params"         -> np.ndarray      Parameters of the benchmark model
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
                    U: Literal["Exact", "Trotter"] = "Exact",
                    benchm: Optional[AbstractModel] = None,
                    params: Optional[np.ndarray] = None,
                    batch_wavefunctions: Optional[np.ndarray] = None,
                    sample_size: int = -1):
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

        if batch_wavefunctions is None:
            self.batch_size = 0
            self.batch_wavefunctions = [None, None]
        else:
            assert(np.size(batch_wavefunctions[0,:]) == 2**np.size(model.qubits)),\
                "Dimension of given batch_wavefunctions do not fit to provided model; n from wf: {}, n qubits: {}".\
                    format(np.log2(np.size(batch_wavefunctions[0,:])), np.size(model.qubits))
            self.batch_size = np.size(batch_wavefunctions[:,0])
            self._generate_batch_wfcts(batch_wavefunctions, U, benchm, params)
            
        self.sample_size = sample_size
        self._U = U
        self._benchm = benchm
        self._params = params
        self._N = 2**np.size(model.qubits)

    def _generate_batch_wfcts(self, initials: np.ndarray, U: Literal["Exact", "Trotter"] = "Exact", benchm: Optional[AbstractModel] = None, params: Optional[np.ndarray]=None):
        self.batch_wavefunctions = [np.copy(initials), np.copy(initials)]
        if(U == "Exact"):
            for k in range(len(initials)):
                self.batch_wavefunctions[1][k] = self._Ut @ initials[k]
        elif(U == "Trotter"):
            assert benchm is not None, "Please provide model for randomized benchmarking"
            assert params is not None, "Please provide parameters for randomized benchmarking"
            for k in range(len(initials)):
                self.batch_wavefunctions[1][k] = benchm.simulator.simulate(benchm.circuit, param_resolver=benchm.get_param_resolver(params), initial_state=initials[k]).state_vector()

    def evaluate(self, wavefunction: np.ndarray, indices: Optional[list] = None) -> np.float64:
        # Here we already have the correct model._Ut
        if self.batch_size == 0 and indices == None:
            #Calculation via Forbenius norm
            #Then the "wavefunction" is the U(t) via VQE
            return 1 - abs(np.trace(np.matrix.getH(self._Ut) @ wavefunction)) / self._N
        else:
            assert (self.batch_size > 0), 'Please provide batch wavefunction'
            assert type(indices) is not None, 'Please provide indices for batch'
            #Calculation via randomly choosing one state vector
            cost=0
            for k in range(len(indices)):
                #This assumes the batch to be the outputs of self._Ut (more efficient if reused as a traning data set)
                cost = cost + 1 - abs( np.matrix.getH(wavefunction[k]) @ self.batch_wavefunctions[1][indices[k]])
            return 1/len(indices) * cost

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
                "U": self._U,
                "benchm": self._benchm,
                "params": self._params,
                "batch_wavefunctions": self.batch_wavefunctions[0],
                "sample_size": self.sample_size
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)
