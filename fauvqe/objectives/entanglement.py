"""
    Implementation of the entanglement entropy as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional, Union, List
from numbers import Integral, Real

import numpy as np
from scipy.linalg import logm, fractional_matrix_power

from qutip import Qobj

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
#from fauvqe import Objective, AbstractModel
import cirq

class Entanglement(Objective):
    """
    Entanglement objective

    This class implements as objective the entanglement of two subspaces of a state given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "alpha"     -> np.float64            Renyi-index. An Index of 1 indicates the use of von Neumann entropy
                "indices"   -> Optional[List[int]]        Subsystem, for which the entanglement entropy shall be calculated
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Entanglement Renyi index=self._alpha>
    
    evaluate(self, wavefunction, indices) : np.float64
        Returns
        ---------
        np.float64:
            Entanglement Entropy of the subsystem indicated by _indices_
    """
    def __init__(   self,
                    model: AbstractModel, 
                    alpha: np.float64 = 1,
                    indices: Optional[List[int]] = None):
        
        self._alpha = alpha
        if(indices is None):
            self._indices = range(int(np.size(model.qubits) / 2 ))
        else:
            self._indices = indices
        
        super().__init__(model)
        self._n = np.size(model.qubits)

    def evaluate(self, wavefunction: Union[np.ndarray, Qobj]) -> np.float64:
        if(isinstance(wavefunction, np.ndarray)):
            q = Qobj(wavefunction, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        elif(isinstance(wavefunction, Qobj)):
            q = wavefunction
        else:
            assert False, 'Please provide either np.ndarray or qutip.Qobj'
        rho = q.ptrace(self._indices)
        #Use von Neumann Entropy for alpha = 1 and Renyi entropy else
        if(self._alpha == 1):
            return np.real( - np.trace(rho * logm(rho)) )
        else:
            return np.real( 1/(1-self._alpha) * np.log(np.trace(fractional_matrix_power(rho, self._alpha))))
        
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "alpha": self._alpha,
                "indices": self._indices
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Entanglement Renyi index={}>".format(self._alpha)