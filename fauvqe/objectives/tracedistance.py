"""
    Implementation of the trace distance as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional, Union
from numbers import Integral, Real

import numpy as np

import qutip
import qutip.metrics

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
import cirq

class TraceDistance(Objective):
    """
    Trace Distance objective

    This class implements as objective the trace distance of two states given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "target"    -> np.ndarray    target state to calculate trace distance to
                
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Trace Distance target=self.target>
    
    evaluate(self, wavefunction) : np.float64
        Returns
        ---------
        np.float64:
            qutip.metrics.tracedist(wavefunction, self.target)
    """
    def __init__(   self,
                    model: AbstractModel, 
                    target: Union[np.ndarray, qutip.Qobj]):
        
        self._N = 2**np.size(model.qubits)
        self._n = np.size(model.qubits)
        if(not isinstance(target, qutip.Qobj)):
            self._target_state = qutip.Qobj(target, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        else:
            self._target_state = target
        
        super().__init__(model)
        
    def evaluate(self, wavefunction) -> np.float64:
        if(isinstance(wavefunction, np.ndarray)):
            q = qutip.Qobj(wavefunction, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        elif(isinstance(wavefunction, qutip.Qobj)):
            q = wavefunction
        else:
            assert False, "Please provide target state as np.ndarray or qutip.Qobj"
        return qutip.metrics.tracedist(q, self._target_state)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "target": self._target_state
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Trace Distance target={}>".format(self._target_state)