"""
    Implementation of the fidelity as objective function for an AbstractModel object.
"""
from numpy import float64 as np_float64
from numpy import ndarray as np_ndarray
from numpy import size as np_size 

from cirq import fidelity as cirq_fidelity
from typing import Dict

from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.objective import Objective

class Fidelity(Objective):
    """
    Fidelity objective

    This class implements as objective the fidelity of two states given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "target_state"    -> np_ndarray    target state to calculate fidelity with

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Fidelity target state=self._target_state>

    evaluate(self, wavefunction) : np_float64
        Returns
        ---------
        np_float64:
    """
    def __init__(   self,
                    model: AbstractModel, 
                    target_state: np_ndarray):
        super().__init__(model)
        self._n = np_size(model.qubits)
        self.set_target_state(target_state)

    def set_target_state(   self, 
                            target_state: np_ndarray) -> None:
        self._target_state = target_state
    
    def evaluate(   self, 
                    wavefunction: np_ndarray,
                    target_state: np_ndarray = None) -> np_float64:
        if target_state is None:
            target_state = self._target_state

        #if(np_size(wavefunction) == 2**self._n):
        #    return abs(wavefunction.transpose() @ self._target_state.full().conjugate())
        #elif(np_size(wavefunction) == 4**self._n):
        #    return abs(self._target_state.full().transpose().conjugate() @ wavefunction @ self._target_state.full())
        #else:
        assert (np_size(wavefunction) == 2**self._n or np_size(wavefunction) == 4**self._n),\
            "State vector (2**self._n = {}) or density matrix (4**self._n = {}) expected; received dimensions: {}"\
                .format(2**self._n, 4**self._n, np_size(wavefunction))
        return cirq_fidelity(target_state, wavefunction, qid_shape=(2,) * self._n)



    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "target_state": self._target_state,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Fidelity target_state={}>".format(self._target_state)
