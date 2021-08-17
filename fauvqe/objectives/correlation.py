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
    qreg: List[cirq.ops.Qid]        list of qubits on which n-point correlation shall be calculated
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Correlation field=self.field>
    """

    def __init__(self, model: AbstractModel, field: Union[Literal["X", "Y", "Z"], cirq.PauliString] = "Z", qreg: List[cirq.ops.Qid]=[cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]):
        self._field = field
        self._qreg = qreg
        if(isinstance(field, cirq.PauliString)):
            super().__init__(model, field)
            return
        elif(field == "X"):
            obs = cirq.X
        elif(field == "Y"):
            obs = cirq.Y
        elif(field == "Z"):
            obs = cirq.Z
        else:
            assert False, "Please choose a Pauli matrix or provide cirq.PauliString for the n-point correlation"
        super().__init__(model, cirq.PauliString(obs(qubit) for qubit in qreg))
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "field": self._field,
                "qreg": self._qreg
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Correlation field={}>".format(self._field)