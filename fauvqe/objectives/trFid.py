"""
    Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional
from numbers import Integral, Real

import numpy as np

import qutip
import qutip.metrics

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
import cirq

class TrFid(Objective):
    """
    Trace Distance objective

    This class implements as objective the trace distance of two states given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "target"    -> np.ndarray    target state to calculate fidelity with
                
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
                    target):
        
        self._N = 2**np.size(model.qubits)
        self._n = np.size(model.qubits)
        if(not isinstance(target, qutip.Qobj)):
            self.target = qutip.Qobj(target, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        else:
            self.target = target
        
        super().__init__(model)
        
    def evaluate(self, wavefunction) -> np.float64:
        if(isinstance(wavefunction, np.ndarray)):
            q = qutip.Qobj(wavefunction, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        elif(isinstance(wavefunction, qutip.Qobj)):
            q = wavefunction
        else:
            raise NotImplementedError()
        return qutip.metrics.tracedist(q, self.target)

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "target": self.target
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Trace Distance target={}>".format(self.target)