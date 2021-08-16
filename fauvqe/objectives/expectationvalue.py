"""Implementation of the expectation value as objective function for an AbstractModel object.
"""
from typing import Literal, Tuple, Dict
from numbers import Integral

import numpy as np

from fauvqe.objectives.objective import Objective
from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.absexpectationvalue import AbsExpectationValue


class ExpectationValue(AbsExpectationValue):
    """Energy expectation value objective

    This class implements as objective the expectation value of the energies
    of the linked model.

    Parameters
    ----------
    model: AbstractModel    The linked model
    field: {"X", "Z"}, default "Z"    The field to be evaluated

    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <Energy field=self.field>
    """

    def __init__(self, model: AbstractModel, field: Literal["Z", "X"] = "Z"):
        super().__init__(model, model.hamiltonian)
        assert field in [
            "Z",
            "X",
        ], "Bad argument 'field'. Allowed values are ['X', 'Z' (default)], received {}".format(
            field
        )

        self.__field: Literal["Z", "X"] = field
        self.__energies: Tuple[np.ndarray, np.ndarray] = model.energy()
        self.__n_qubits: Integral = np.log2(np.size(self.__energies[0]))
    
    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        if self.__field == "X":
            wf_x = self._rotate_x(wavefunction)

            return (
                np.sum(
                    np.abs(wavefunction) ** 2 * (-self.__energies[0])
                    + np.abs(wf_x) ** 2 * (-self.__energies[1])
                )
                / self.__n_qubits
            )

        # field must be "Z"
        return (
            np.sum(np.abs(wavefunction) ** 2 * (-self.__energies[0] + self.__energies[1]))
            / self.__n_qubits
        )

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "field": self.__field,
                "model": self._model,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<Energy field={}>".format(self.__field)
