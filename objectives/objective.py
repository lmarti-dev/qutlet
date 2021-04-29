import numpy as np
from abc import abstractmethod

class Objective:
    def __init__(self):
        # Add np.ndarray type to self.__energies
        self.__energies = None

    def initialise(self, energies: np.ndarray) -> None:
        self.__energies = energies

    @abstractmethod
    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        raise NotImplementedError()
