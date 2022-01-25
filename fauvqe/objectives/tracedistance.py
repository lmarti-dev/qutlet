"""
    Implementation of the trace distance as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional, Union
from numbers import Integral, Real

import numpy as np

from qutip import Qobj
from qutip.metrics import tracedist

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
    options:    "target_state"    -> np.ndarray    target state to calculate trace distance to
                
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
            qutip.metrics.tracedist(wavefunction, self._target_state)
    """
    def __init__(   self,
                    model: AbstractModel, 
                    target_state: Union[np.ndarray, Qobj]):
        super().__init__(model)
        self._N = 2**np.size(model.qubits)
        self._n = np.size(model.qubits)
        self.set_target_state(target_state)
    
    def set_target_state(self, target_state: Union[np.ndarray, Qobj]) -> None:
        if(not isinstance(target_state, Qobj)):
            self._target_state = Qobj(target_state, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        else:
            self._target_state = target_state
    
    def evaluate(self, wavefunction) -> np.float64:
        if(isinstance(wavefunction, np.ndarray)):
            return tracedist(
                Qobj(wavefunction, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]]), 
                self._target_state
            )
        elif(isinstance(wavefunction, Qobj)):
            return tracedist(wavefunction, self._target_state)
        else:
            assert False, "Please provide target state as np.ndarray or qutip.Qobj"

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "target_state": self._target_state
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Trace Distance target_state={}>".format(self._target_state)