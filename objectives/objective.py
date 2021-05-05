import numpy as np
from typing import Tuple
from abc import abstractmethod, ABC
import cirq

from ..isings.initialisers import Initialiser


class Objective(ABC):
    def __init__(self):
        # Add np.ndarray type to self.__energies
        self.__initialiser: Initialiser = None

    def initialise(self, initialiser: Initialiser) -> None:
        self.__initialiser = initialiser

    @abstractmethod
    def evaluate(self, wavefunction: np.ndarray) -> np.float64:
        raise NotImplementedError()

    def _rotate_x(self, wavefunction: np.ndarray) -> np.ndarray:
        rotation_circuit = cirq.Circuit()
        for row in self.__initialiser.qubits:
            for qubit in row:
                rotation_circuit.append(cirq.H(qubit))

        return self.__initialiser.simulator.simulate(
            rotation_circuit, initial_state=wavefunction
        ).state_vector()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
