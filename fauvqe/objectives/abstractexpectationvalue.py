"""Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Mapping, Optional
from numbers import Integral

import numpy as np
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel


class AbstractExpectationValue(Objective):
    """Abstract Expectation value objective

    This class implements as objective the expectation value of a given observable.

    Parameters
    ----------
    model: AbstractModel        The linked model
    observable: Optional[cirq.PauliSum]       Observable of which the expectation value will be calculated. If not provided, the function evaluate needs to be overwritten
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <AbstractExpectationValue field=self.field>
    """

    def __init__(self, model: AbstractModel, observable: Optional[cirq.PauliSum]):
        self._observable = observable
        super().__init__(model)
        
    def evaluate(self, wavefunction: np.ndarray, q_map, atol: float = 1e-7) -> np.float64:
        if(self._observable is not None):
            return self._observable.expectation_from_state_vector(wavefunction, q_map, atol=atol)
        else:
            raise NotImplementedError()

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