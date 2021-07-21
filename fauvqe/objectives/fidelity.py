"""
    Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional
from numbers import Integral, Real

import numpy as np

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
    options:    "t"         -> Float    t in U(t)
                "U"         -> Literal  exact or Trotter (only matters if not exists or t wrong)
                "wavefunctions"  -> np.ndarray      if None Use U csot, 
                                                    otherwise batch wavefunctions for random batch cost
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
                    target: np.ndarray,
                    pure: bool = False):
        
        self.target = target
        self.pure = pure
        
        super().__init__(model)
        self._N = 2**np.size(model.qubits)

    def evaluate(self, wavefunction) -> np.float64:
    if(self.pure):
        assert(isinstance(wavefunction, np.ndarray) and isinstance(self.target, np.ndarray)), 'WARNING: Input not of type np.ndarray'
        return abs(np.matrix.getH(vwavefunction) @ self.target)
    else:
        assert(isinstance(wavefunction, qutip.qobj) and isinstance(self.target, qutip.qobj)), 'WARNING: Input not of type qutip.qobj'
        return fidelity(wavefunction, self.target)

    def to_json_dict(self) -> Dict:
        raise NotImplementedError() 
        return {
            "constructor_params": {
                "model": self._model,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        raise NotImplementedError() 
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<UtCost t={}>".format(self.t)
