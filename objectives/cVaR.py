from .objective import Objective
import numpy as np

class cVaR(Objective):
    def __init__(self, alpha: float):
        super().__init__()

        assert (
            alpha <= 1.0 and alpha >= 0
        ), "cVaR alpha must be in (0, 1). Recieved: {:f}".format(alpha)

        self.__alpha = alpha
        self.__mask = None

    def initialise(self, energies: np.ndarray) -> None:
        self.__mask = np.argsort(energies)

        self.__energies = energies[self.__mask]

    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        # Wavefunction is expected to be normalized
        # Reorder wavefunction with stored mask
        probabilities = np.abs(wavefunction)**2
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

        return cvar

