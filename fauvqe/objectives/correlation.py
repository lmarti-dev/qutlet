"""Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict, Mapping, Optional
from numbers import Integral

import numpy as np
import cirq

from fauvqe.objectives.objective import Objective
from fauvqe.objectives.expectationvalue import AbsExpectationValue
from fauvqe.models.abstractmodel import AbstractModel


class Correlation(AbsExpectationValue):
    """Correlation Expectation value objective

    This class implements as objective the correlation of a given state vector.

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

    def __init__(self, model: AbstractModel, field: Literal["X", "Y", "Z"] = "Z", rows: list = [0,0], cols: list = [0,1]):
        assert len(rows) == len(cols), 'Please provide as many row indices as column indices for cirq.GridQubit'
        if(field == "X"):
            obs = cirq.X
        elif(field == "Y"):
            obs = cirq.Y
        elif(field == "Z"):
            obs = cirq.Z
        else:
            raise NotImplementedError()
        
        self.field = field
        self.rows = rows
        self.cols = cols
        super().__init__(model, cirq.PauliString(obs(model.qubits[rows[k]][cols[k]]) for k in range(len(rows))))
    
    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
                "field": self.field,
                "cols": self.cols,
                "rows": self.rows
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Correlation field={}>".format(self.field)