"""
    Implementation of the magnetisation as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Mapping, Optional
from numbers import Integral

import numpy as np
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
from fauvqe.models.abstractmodel import AbstractModel


class Magnetisation(AbstractExpectationValue):
    """Magnetisation Expectation value objective

    This class implements as objective the magnetisation of a given state vector.

    Parameters
    ----------
    model: AbstractModel        The linked model
    field: Literal["X", "Y", "Z"]        Orientation of Magnetic field to be calculated
    position: list        list of qubit numbers on which magnetisation shall be calculated
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Magnetisation field=self.field>
    """

    def __init__(self, model: AbstractModel, field: Literal["X", "Y", "Z"] = "Z", row: int = 0, col: int = 0):
        self._field = field
        self._row = row
        self._col = col
        if(field == "X"):
            obs = cirq.X
        elif(field == "Y"):
            obs = cirq.Y
        elif(field == "Z"):
            obs = cirq.Z
        else:
            raise NotImplementedError()
        super().__init__(model, cirq.PauliString(obs(model.qubits[row][col])))
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "field": self._field,
                "col": self._col,
                "row": self._row
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Magnetisation field={}>".format(self._field)