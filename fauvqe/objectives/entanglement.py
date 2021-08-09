"""
    Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Optional
from numbers import Integral, Real

import numpy as np
import scipy

import qutip

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
import cirq

class Entanglement(Objective):
    """
    Entanglement objective

    This class implements as objective the entanglement of two subspaces of a state given as a pure state vector or as a density matrix

    Parameters
    ----------
    model: AbstractModel, The linked model
    options:    "pure"      -> bool                            True, if target and input are both pure
                "typ"       -> Literal['Neumann', 'Renyi']     Determines, whether to calculate von Neumann or Renyi Entanglement Entropy
                "alpha"     -> Optional[np.float64]            Renyi-index. Only used, if typ == 'Renyi'
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Fidelity target=self.target>
    
    evaluate(self, wavefunction, indices) : np.float64
        Returns
        ---------
        np.float64:
            Entanglement Entropy of the subsystem indicated by _indices_
    """
    def __init__(   self,
                    model: AbstractModel, 
                    pure: bool = False,
                    typ: Literal['Neumann, Renyi'] = 'Neumann',
                    alpha: Optional[np.float64] = None,
                    indices: Optional[list] = None):
        
        self.pure = pure
        self.typ = typ
        if(typ == 'Renyi'):
            assert alpha is not None, 'Please provide a Renyi index'
        self.alpha = alpha
        if(indices is None):
            self.indices = range(int(np.size(self.model.qubits) / 2 ))
        else:
            self.indices = indices
        
        super().__init__(model)
        self._N = 2**np.size(model.qubits)
        self._n = np.size(model.qubits)

    def evaluate(self, wavefunction) -> np.float64:
        if(self.pure):
            q = qutip.Qobj(wavefunction, dims=[[2 for k in range(self._n)], [1 for k in range(self._n)]])
        else:
            assert isinstance(wavefunction, qutip.Qobj), 'Please provide a qutip Qobj'
            q = wavefunction
        rho = q.ptrace(self.indices)
        if(self.typ == 'Neumann'):
            return np.real( - np.trace(rho * scipy.linalg.logm(rho)) )
        elif(self.typ == 'Renyi'):
            return np.real( 1/(1-self.alpha) * np.log(np.trace(scipy.linalg.fractional_matrix_power(rho, self.alpha))))
        else:
            raise NotImplementedError()

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "pure": self.pure,
                "typ": self.typ,
                "alpha": self.alpha,
                "indices": self.indices
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Entanglement type={}>".format(self.typ)