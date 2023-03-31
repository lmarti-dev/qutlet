"""
    Abstract base class for Objectives required to optimise circuit.
"""
import abc
from numbers import Real
from typing import Optional, List

import numpy as np
import cirq

from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.restorable import Restorable


class Objective(Restorable):
    """Abstract base class for Objectives required to optimise circuit.

    This class is unusable and intended to be extended.
    An implementation of this type **must** extend the methods `evaluate`.
    Furthermore, it **must** implement the python magic method `__repr__`
    in a specific format given below.
    Due to the hierarchy of this class, it is necessary to keep objective
    specific arguments, i.e. a potential field choice, in the constructor of subclasses.

    For an example of how to implement this, refer to ExpectationValue or CVaR.

    Methods
    ----------
    __repr__() : str
        <CLASSNAME constructorparam1=val constructorparam2=val>

    """

    def __init__(self, model: AbstractModel):
        assert isinstance(
            model, AbstractModel
        ), "Bad argument 'model'. Must be an instance of AbstractModel. '{}' was given".format(
            type(model).__name__
        )

        self._model: AbstractModel = model

    @property
    def model(self) -> AbstractModel:
        """The AbstractModel instance linked to this objective

        Returns
        -------
        AbstractModel
        """
        return self._model

    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Simulate the circuit of the model with a given parameter resolver.

        Parameters
        ----------
        param_resolver
            The circuit parameters (consider generating with model.get_param_resolver())

        initial_state: numpy.ndarray, optional
            The initial wavefunction to start the simulation with

        Returns
        -------
        numpy.ndarray: The simulated wavefunction.
        """
        wf = self._model.simulator.simulate(
            self._model.circuit,
            param_resolver=param_resolver,
            initial_state=initial_state,
        ).state_vector()

        return wf / np.linalg.norm(wf)

    @abc.abstractmethod
    def evaluate(self, wavefunction: np.ndarray) -> Real:
        """Calculate the objective for a given wavefunction.

        Parameters
        ----------
        wavefunction: numpy.ndarray
            The wavefunction to calculate the corresponding objective.

        Returns
        ----------
        Real:
            The value of the objective function for the given wavefunction.
        """
        raise NotImplementedError()  # pragma: no cover

    def _rotate_x(self, wavefunction: np.ndarray) -> np.ndarray:
        """Helper method to rotate a wavefunction along the x axis.

        Uses a rotation cirq circuit constructed from Hadamard gates to perform the transformation.

        Parameters
        ---------
        wavefunction: numpy.ndarray
            Wavefunction to rotate around the x axis

        Returns
        ---------
        numpy.ndarray
            Rotated wavefunction of the same size as the input wavefunction
        """
        return self._rotate(wavefunction, ["X" for k in range(self._model.n[0] * self._model.n[1])])

    def _rotate_y(self, wavefunction: np.ndarray) -> np.ndarray:
        return self._rotate(wavefunction, ["Y" for k in range(self._model.n[0] * self._model.n[1])])

    def _rotate(self, wavefunction: np.ndarray, bases: List[str]) -> np.ndarray:
        assert len(wavefunction) == 2 ** (
            len(bases)
        ), "List of bases does not fit dimension of wavefunction"
        rotation_circuit = cirq.Circuit()
        hadamard = lambda q: cirq.H(q)
        s_dagger = lambda q: cirq.Z(q) ** (3 / 2)
        i = 0
        for row in self._model.qubits:
            for qubit in row:
                if bases[i] == "X":
                    rotation_circuit.append(hadamard(qubit))
                elif bases[i] == "Y":
                    rotation_circuit.append(s_dagger(qubit))
                    rotation_circuit.append(hadamard(qubit))
                elif bases[i] == "Z":
                    pass
                else:
                    raise NotImplementedError()
                i += 1

        return self._model.simulator.simulate(
            rotation_circuit,
            # Start off at the given wavefunction
            initial_state=wavefunction,
        ).state_vector()

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()  # pragma: no cover
