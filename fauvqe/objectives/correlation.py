"""
    Implementation of the n-point correlation expectation value as objective function for an AbstractModel object. 
"""
from typing import Literal, Tuple, Dict, Mapping, Optional, List, Union
from numbers import Integral

import numpy as np
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
from fauvqe.models.abstractmodel import AbstractModel


class Correlation(AbstractExpectationValue):
    """Correlation Expectation value objective

    This class implements as objective the n-point correlation of a given state vector. The special case n=1 reduces to the magnetisation objective.

    Parameters
    ----------
    model: AbstractModel        The linked model
    field: Literal["X", "Y", "Z"]        Pauli Matrix to be used in n-point correlation
    points: List[cirq.ops.Qid]        list of qubits on which n-point correlation shall be calculated
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Correlation field=self.field>
    """

    def __init__(self, model: AbstractModel, field: Union[Literal["X", "Y", "Z"], cirq.PauliString] = "Z", points: List[cirq.ops.Qid]=[cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]):
        self._field = field
        self._points = points
        if(isinstance(field, Union[cirq.PauliSum, cirq.PauliString])):
            super().__init__(model, field)
            return
        elif(field == "X"):
            self._pauli = cirq.X
        elif(field == "Y"):
            self._pauli = cirq.Y
        elif(field == "Z"):
            self._pauli = cirq.Z
        else:
            assert False, "Please choose a Pauli matrix or provide cirq.PauliString for the n-point correlation"
        super().__init__(model, cirq.PauliString(self._pauli(qubit) for qubit in points))
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "field": self._field,
                "points": self._points
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Correlation observable={}>".format(self._observable)