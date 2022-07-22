"""
    Implementation of the fidelity as objective function for an AbstractModel object.
"""
from numbers import Integral, Real
import numpy as np
from qutip.metrics import fidelity as qutip_fidelity
from qutip import Qobj as qutip_Qobj
from typing import Dict, Literal, Optional, Tuple, Union

from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.objective import Objective


class Fidelity(Objective):
    """
    Fidelity objective

    This class implements as objective the fidelity of two states given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "target_state"    -> np.ndarray    target state to calculate fidelity with
                
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Fidelity target state=self._target_state>
    
    evaluate(self, wavefunction) : np.float64
        Returns
        ---------
        np.float64:
            if(self.pure):
                target^dagger @ wavefunction
            else:
                qutip.metrics.fidelity(target, wavefunction)
    """
    def __init__(   self,
                    model: AbstractModel, 
                    target_state: Union[np.ndarray, qutip_Qobj]):
        super().__init__(model)
        self._n = np.size(model.qubits)
        self.set_target_state(target_state)

    def set_target_state(self, target_state: Union[np.ndarray, qutip_Qobj]) -> None:
        if(not isinstance(target_state, qutip_Qobj)):
            self._target_state = qutip_Qobj(target_state, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        else:
            self._target_state = target_state
    
    def evaluate(self, wavefunction: Union[np.ndarray, qutip_Qobj]) -> np.float64:
        if(isinstance(wavefunction, np.ndarray)):
            #q = qutip_Qobj(wavefunction, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
            if(np.size(wavefunction) == 2**self._n):
                return abs(wavefunction.transpose() @ self._target_state.full().conjugate())
            elif(np.size(wavefunction) == 2**(2*self._n)):
                return abs(self._target_state.full().transpose().conjugate() @ wavefunction @ self._target_state.full())
            else:
                assert False, "State vector or density matrix expected got dimensions: {}".format(np.size(wavefunction))
        elif(isinstance(wavefunction, qutip_Qobj)):
            q = wavefunction
        else:
            raise NotImplementedError()
        return qutip_fidelity(q, self._target_state)

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
        return "<Fidelity target_state={}>".format(self._target_state)