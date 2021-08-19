"""
    Abstract class for the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Mapping, Optional
from numbers import Integral

import numpy as np
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel


class AbstractExpectationValue(Objective):
    """Abstract Expectation value objective
    
    This class implements an abstract class for expectation values of a given observable.
    
    Parameters
    ----------
    model: AbstractModel        The linked model
    observable: Optional[cirq.PauliSum]       Observable of which the expectation value will be calculated. If not provided, the model hamiltonian will be set
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <AbstractExpectationValue observable=self._observable>
    """

    def __init__(self, model: AbstractModel, observable: Optional[cirq.PauliSum]=None):
        super().__init__(model)
        if(observable is None):
            self._observable = model.hamiltonian
        else:
            self._observable = observable
    
    def evaluate(self, wavefunction: np.ndarray, q_map: Mapping[cirq.ops.pauli_string.TKey, int]=None, atol: float = 1e-7) -> np.float64:
        if(q_map is None):
            q_map = {self._model.qubits[k][l]: int(k*self._model.n[1] + l) for l in range(self._model.n[1]) for k in range(self._model.n[0])}
        if(wavefunction.ndim == 1):
            return self._observable.expectation_from_state_vector(wavefunction, q_map, atol=atol)
        elif(wavefunction.ndim==2):
            return self._observable.expectation_from_density_matrix(wavefunction, q_map, atol=atol)
        else:
            assert False, 'Please provide either state vector or density matrix'
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "observable": self._observable
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<AbstractExpectationValue observable={}>".format(self._observable)