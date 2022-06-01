"""
    This abstract class is a generalisation of Objective and allows to use combination of Objectives
"""
import abc
from numbers import Real
from typing import List, Optional, Union

#Change to from cirq import
import cirq
import numpy as np


from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.objective import Objective
from fauvqe.restorable import Restorable


class ObjectiveSum(Restorable):
    #TODo rewrite docstring
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
    #TODO 1
    # allow for an arbitrary smpy? combined_objective_fct
    #Idea: all objectives return some values x_1 ,...x_m 
    #Then combined_objective_fct is a function f(x_1 ,...x_m) -> Reals
    #TODO 2
    #   Overwrite +, -, * and /
    #   To allow 0.3*ExpectionValue(ising) - Entanglement(ising)/2
    #   To return the correct ObjectiveSum
    def __init__(   self, 
                    models: Union[List[AbstractModel], AbstractModel],
                    objectives: Union[List[Objective], Objective],
                    combined_objective_fct: None):
        assert (len(models) == len(objectives)), 
        "ObjectiveSum error: number of provided models and objectives must be equal. len(models) != len(objectives): {} != {}".format(
            len(models), len(objectives))
        #Maybe model not even needed here?
        self._models: Union[List[AbstractModel], AbstractModel] = models
        self._objectives: Union[List[Objective], Objective] = objectives
        self._combined_objective_fct= combined_objective_fct

    @property
    def models(self) -> AbstractModel:
        """The AbstractModel instance linked to this objective

        Returns
        -------
        AbstractModel
        """
        return self._models

    def simulate(self, param_resolver, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        #TODO: Allow for sampling and sampling budget allocation
        #Return Union[List[nd.array], nd.array]
        #Note that for both evaluate and simulate, simualte and evaluate of the respective objective is used
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

        return wf/np.linalg.norm(wf)

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
        assert len(wavefunction) == 2**(len(bases)), "List of bases does not fit dimension of wavefunction"
        rotation_circuit = cirq.Circuit()
        hadamard = lambda q: cirq.H(q)
        s_dagger = lambda q: cirq.Z(q)**(3/2)
        i=0
        for row in self._model.qubits:
            for qubit in row:
                if(bases[i] == "X"):
                    rotation_circuit.append(hadamard(qubit))
                elif(bases[i] == "Y"):
                    rotation_circuit.append(s_dagger(qubit))
                    rotation_circuit.append(hadamard(qubit))
                elif(bases[i] == "Z"):
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
        """
            Representation function.

            Use here the __repr__ function of the models and objectives themselves

    
        """
        repr_string= "<ObjectiveSum:\n"

        return "<ObjectiveSum: {}>".format(self.__energy_fields)