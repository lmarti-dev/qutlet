"""
module docstring Optimisation result
"""
from typing import List
from numbers import Real

import numpy as np

from ..objectives import Objective
from .optimisation_step import OptimisationStep
from .optimiser import Optimiser


class OptimisationResult:
    """
    class docstring Optimisation result
    """

    def __init__(self, optimiser: Optimiser):
        # Store start params? optimiser.circuit_param_values
        self.__steps: List[OptimisationStep] = []
        self.__wavefunctions: np.ndarray = None
        self.__optimiser: Optimiser = optimiser

    def add_step(self, step: OptimisationStep) -> None:
        """
        Add a step
        """
        self.__steps.append(step)

    def get_steps(self) -> List[OptimisationStep]:
        """
        Get all steps
        """
        return self.__steps

    def get_latest_step(self) -> OptimisationStep:
        """
        Get latest/last step
        """
        return self.__steps[-1]

    def finalize(self) -> None:
        """
        Finalize (calc wavefunction, joblib here!)
        """
        self.__wavefunctions = self.get_wavefunctions()

    def get_wavefunctions(self) -> List[np.ndarray]:
        """
        Get wavefunctions
        """
        if self.__wavefunctions is not None:
            return self.__wavefunctions

        return [self.__get_wavefunction(step.params) for step in self.__steps]

    def get_latest_wavefunction(self) -> np.ndarray:
        """
        Get latest/last wavefunction
        """

        if self.__wavefunctions is not None:
            return self.__wavefunctions[-1]

        return self.__get_wavefunction(self.get_latest_step().params)

    def get_objective(self) -> Objective:
        """
        Get objective used for optimisation
        """

        return self.__optimiser.objective

    def get_objective_values(self) -> List[Real]:
        """
        Get value of the objective.
        """

        return [self.__optimiser.objective.evaluate(wf) for wf in self.get_wavefunctions()]

    def get_latest_objective_value(self):
        """
        Get final value of the objective.
        """

        final_step = self.get_latest_step()
        wavefunction = self.__get_wavefunction(final_step.params)

        return self.__optimiser.objective.evaluate(wavefunction)

    def __get_wavefunction(self, params: np.ndarray) -> np.ndarray:
        """
        Helper method to simulate the wavefunction.
        """

        return self.__optimiser.simulator.simulate(
            self.__optimiser.circuit,
            param_resolver=self.__optimiser._get_param_resolver(params),
        ).state_vector()

    def __repr__(self) -> str:
        return "<OptimisationResult steps={} last_step={} result={} objective={}>".format(
            len(self.__steps),
            self.get_latest_step(),
            self.get_latest_objective_value(),
            self.__optimiser.objective,
        )
