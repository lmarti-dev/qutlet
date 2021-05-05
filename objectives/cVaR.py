import numpy as np
from typing import Literal

from .objective import Objective
from ..isings.initialisers import Initialiser


class cVaR(Objective):
    def __init__(self, alpha: float):
        super().__init__()

        assert alpha <= 1.0 and alpha >= 0, "cVaR alpha must be in (0, 1). Recieved: {:f}".format(alpha)

        self.__alpha = alpha
        self.__mask = None
        self.__energies: np.ndarray = None

    def initialise(self, initialiser: Initialiser, field: Literal["Z", "X"]) -> None:
        self.__initialiser = initialiser

        assert field in [
            "Z",
            "X",
        ], "Bad input 'field'. Allowed values are ['X', 'Z' (default)], revieced {}".format(field)
        self.__field = field

        energies = self.__initialiser.energy()
        energies = -energies[0] + energies[1]
        self.__mask = np.argsort(energies)

        self.__energies = energies[self.__mask]

    def evaluate(self, wavefunction: np.ndarray) -> np.float64:

        # Wavefunction is expected to be normalized
        # Reorder wavefunction with stored mask
        probabilities = np.abs(wavefunction) ** 2
        probabilities = probabilities[self.__mask]

        # Only use until total probability adds up to alpha
        mask = np.cumsum(probabilities) <= self.__alpha

        # First value is cVaR
        if not np.any(mask):
            return self.__energies[0]

        # Apply mask
        energies_ = self.__energies[mask]
        probabilities_ = probabilities[mask]

        # sum of probability times value over sum of probabilities
        cvar = np.sum(energies_ * probabilities_) / np.sum(probabilities_)

        print(self.__initialiser.qubits, len(self.__initialiser.qubits))

        return cvar / len(self.__initialiser.qubits)

    def __repr__(self) -> str:
        return "<cVaR field={} alpha={}>".format(self.__field, self.__alpha)
