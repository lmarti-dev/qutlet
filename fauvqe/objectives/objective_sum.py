"""
    This abstract class is a generalisation of Objective and allows to use combination of Objectives
    TODO Write tests see 20220601.py
"""
import abc
from numbers import Real
from typing import Dict, List, Optional, Union

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
                    objectives: Union[List[Objective], Objective],
                    combined_objective_fct = None):
        self._objectives: Union[List[Objective], Objective] = objectives
        self._combined_objective_fct= combined_objective_fct

    @property
    def objectives(self) -> Union[List[Objective], Objective]:
        """
        The Objectives linked to this ObjectiveSum

        Returns
        -------
            Union[List[Objective], Objective]
        """
        return self._objectives

    def simulate(self, 
                param_resolver, 
                initial_state: Optional[np.ndarray] = None) -> Union[List[np.array], np.array]:
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
        
        #TODO Potentially parallise this
        if isinstance(self._objectives, List):
            wavefunctions = []
            for objective in self._objectives:
                #print(objective)
                wavefunctions.append(objective.simulate(
                        param_resolver=param_resolver,
                        initial_state=initial_state))
            return wavefunctions
        else:
            return self._objectives.simulate(
                        param_resolver=param_resolver,
                        initial_state=initial_state)

    def evaluate(   self, 
                    wavefunctions: Union[List[np.array], np.array]) -> Real:
        """
        Calculate the objective sum for a given wavefunction/sample 

        Parameters
        ----------
        wavefunction: numpy.ndarray
            The wavefunction to calculate the corresponding objective.

        Returns
        ----------
        Real:
            The value of the objective function for the given wavefunction.
        """
        if isinstance(self._objectives, List):
            _tmp = []
            for i in range(len(self._objectives)):
                _tmp.append(self._objectives[i].evaluate(wavefunctions[i]))
        else:
            _tmp = self._objectives.evaluate(wavefunctions)

        if self._combined_objective_fct is None:
            return np.sum(_tmp)
        else:
            # e.g. have here the lambda function
            # lambda x0, x1: 0.5*x0 + x1**2
            return self._combined_objective_fct(*_tmp)
    
    #TODO Add typing
    def set_linear_combined_objective_fct(self, coefficents):
        self._combined_objective_fct = lambda *args: sum([coefficents[i]*args[i] for i in range(len(coefficents))])

    def __repr__(self) -> str:
        """
            Representation function.

            Use here the __repr__ function of the models and objectives themselves    
        """
        repr_string= "<ObjectiveSum:\n"
        for objective in self._objectives:
            repr_string += repr(objective)
        return repr_string

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "objectives": self._objectives,
                "combined_objective_fct": self._combined_objective_fct,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        inst = cls(**dct["constructor_params"])   
        return inst