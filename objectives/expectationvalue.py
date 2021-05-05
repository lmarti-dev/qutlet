from typing import Tuple, Literal, Optional
import numpy as np

from ..isings.initialisers import Initialiser
from .objective import Objective


class ExpectationValue(Objective):
    def __init__(self):
        super().__init__()
        self.__field = "Z"
        self.__initialiser: Initialiser = None

    def initialise(self, initialiser: Initialiser, field: Literal["Z", "X"]) -> None:
        # Type checking
        assert isinstance(
            initialiser, Initialiser
        ), "Bad argument 'initiliser'. Must be an instance of Initiliser"
        assert field in [
            "Z",
            "X",
        ], "Bad argument 'field'. Allowed values are ['X', 'Z' (default)], revieced {}".format(
            field,
        )

        self.__initialiser = initialiser
        self.__energies = self.__initialiser.energy()
        self.__field = field

    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        if self.__field == "X":
            wf_x = self._rotate_x(wavefunction)

            return np.sum(
                np.abs(wavefunction) ** 2 * (-self.__energies[0])
                + np.abs(wf_x) ** 2 * (-self.__energies[1])
            ) / len(self.__initialiser.qubits)
        elif self.__field == "Z":
            return np.sum(
                np.abs(wavefunction) ** 2 * (-self.__energies[0] + self.__energies[1])
            ) / len(self.__initialiser.qubits)

    def __repr__(self) -> str:
        return "<ExpectationValue field={}>".format(self.__field)
