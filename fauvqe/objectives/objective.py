"""Abstract base class for Objectives required to optimise circuit.
"""
import abc
from numbers import Real

import numpy as np
import cirq

from fauvqe.initialisers.initialiser import Initialiser


class Objective(abc.ABC):
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

    def __init__(self, initialiser: Initialiser):
        assert isinstance(
            initialiser, Initialiser
        ), "Bad argument 'initialiser'. Must be an instance of Initialiser. '{}' was given".format(
            type(initialiser).__name__
        )

        self._initialiser: Initialiser = initialiser

    @property
    def initialiser(self) -> Initialiser:
        return self._initialiser

    def simulate(self, param_resolver, initial_state=None):
        simulator_result = self._initialiser.simulator.simulate(
            self._initialiser.circuit,
            param_resolver=param_resolver,
            initial_state=initial_state,
        )

        return simulator_result.state_vector()

    def simulate_and_evaluate(self, param_resolver, initial_state=None):
        return self.evaluate(self.simulate(param_resolver, initial_state=initial_state))

    @abc.abstractmethod
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
        for row in self._initialiser.qubits:
            for qubit in row:
                # Hadamard gate corresponds to x rotation
                rotation_circuit.append(cirq.H(qubit))

        return self._initialiser.simulator.simulate(
            rotation_circuit,
            # Start off at the given wavefunction
            initial_state=wavefunction,
        ).state_vector()

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
