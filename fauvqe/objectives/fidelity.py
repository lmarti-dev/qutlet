"""
    Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional
from numbers import Integral, Real

import numpy as np

import qutip
from qutip.metrics import fidelity

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
import cirq

class Fidelity(Objective):
    """
    Fidelity objective

    This class implements as objective the fidelity of two states given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "target"    -> np.ndarray    target state to calculate fidelity with
                "pure"      -> bool  True, if target and input are both 

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <UtCost field=self.field>
    
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
                    target,
                    pure: bool = False):
        
        self.target = target
        self.pure = pure
        
        super().__init__(model)
        self._N = 2**np.size(model.qubits)

    def evaluate(self, wavefunction) -> np.float64:
        if(self.pure):
            #Numpy's ndarray required
            assert(isinstance(wavefunction, np.ndarray) and isinstance(self.target, np.ndarray)), 'WARNING: Input not of type np.ndarray'
            return abs(np.matrix.getH(wavefunction) @ self.target)
        else:
            #Qutip' Qobj required
            assert(isinstance(wavefunction, qutip.Qobj) and isinstance(self.target, qutip.Qobj)), 'WARNING: Input not of type qutip.qobj'
            return fidelity(wavefunction, self.target)

    def to_json_dict(self) -> Dict:
        raise NotImplementedError() 
        return {
            "constructor_params": {
                "model": self._model,
                "N": self._N,
                "target": self.target,
                "pure": self.pure,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        raise NotImplementedError() 
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Fidelity target={}>".format(self.target)
