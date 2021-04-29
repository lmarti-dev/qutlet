from .objective import Objective
import numpy as np

class ExpectationValue(Objective):
    def __init__(self):
        super().__init__()

    def initialise(self, energies: np.ndarray) -> None:
        self.__energies = energies

    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        assert self.__energies is not None, "ExpectationValue was not initilised. Please run .initilise(energies) first"

        return np.sum(np.abs(wavefunction)**2 * self.__energies) # / n_sites  # Missing n_sites: Is n_sites = shape of wavefunction?
