"""Abstract base class for Objectives required to optimise circuit.
"""
from abc import abstractmethod, ABC
from numbers import Real
from typing import Tuple

import numpy as np
import cirq

from ..isings.initialisers import Initialiser


class Objective(ABC):
    """Abstract base class for Objectives required to optimise circuit.

    This class is unusable and intended to be extended.
    An implementation of this type **must** extend the methods `evaluate`
    and `initialise`
    Furthermore, it **should** implement the python magic method `__repr__`
    in a specific format given below.
    Due to the hierarchy of this class, it is necessary to keep objective
    specific arguments, i.e. a potential field choice, in the constructor of subclasses.

    For an example of how to implement this, refer to ExpectationValue or CVaR.

    Methods
    ----------
    __repr__() : str
        <CLASSNAME constructorparam1=val constructorparam2=val>

    """

    @abstractmethod
    def initialise(self, obj_value: Tuple[np.ndarray]) -> None:
        """Compute one-time tasks from the given objective value.

        This method enables the computation of one-time tasks (i.e. calculating
        energies of the initialiser).
        It should be considered to store the intermediate results in RAM but keep
        in mind how much bandwidth and/or capacity will be used as a result.

        Parameters
        ----------
        obj_value:
            The objective values.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, wavefunction: np.ndarray) -> Real:
        """Calculate the objective for a given wavefunction.

        **Must** be called after `initialise()` as it may use results of computations
        in said method or it may depend on information from the initialiser.

        Parameters
        ----------
        wavefunction:
            The wavefunction to calculate the corresponding objective.

        Returns
        ----------
        Real:
            The value of the objective function for the given wavefunction.

        """
        raise NotImplementedError()

    def _rotate_x(self, wavefunction: np.ndarray) -> np.ndarray:
        """Helper method to rotate a wavefunction around the x axis.

        Uses a rotation cirq circuit constructed from Hadamard gates to perform the transformation.

        Parameters
        ---------
        wavefunction
            Wavefunction to rotate around the x axis

        Returns
        ---------
        numpy.ndarray
            Rotated wavefunction of the same size as the input wavefunction
        """
        # Construct rotation circuit for each qubit
        rotation_circuit = cirq.Circuit()
        for row in self.__initialiser.qubits:
            for qubit in row:
                # Hadamard gate corresponds to x rotation
                rotation_circuit.append(cirq.H(qubit))

        return self.__initialiser.simulator.simulate(
            rotation_circuit,
            # Start off at the given wavefunction
            initial_state=wavefunction,
        ).state_vector()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
