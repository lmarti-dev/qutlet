from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt

from ..objectives import Objective
from .optimisation_step import OptimisationStep
from .optimiser import Optimiser


class OptimisationResult:
    def __init__(self, optimiser: Optimiser):
        # Store start params? optimiser.circuit_param_values
        self.__steps: List[OptimisationStep] = []
        self.__wavefunctions: np.ndarray = None
        self.__optimiser = optimiser

    def savetxt(self, path: str) -> None:
        # Add option to also store wavefunctions/expval/objectives/...
        # Add headers?
        arr = np.array([step.params for step in self.get_steps()])
        np.savetxt(path, arr)

    def readtxt(self, path: str) -> None:
        # Additional classmethod to also recover Ising() ...
        arr = np.loadtxt(path)
        for index in range(len(arr)):
            self.__steps.insert(index, arr[index])

    def add_step(self, step: OptimisationStep) -> None:
        self.__steps.append(step)

    def get_steps(self) -> List[OptimisationStep]:
        return self.__steps

    def get_latest_step(self) -> OptimisationStep:
        return self.__steps[-1]

    def plot(self, objective: Objective) -> np.ndarray:
        objective_values = np.array([objective.evaluate(wf) for wf in self.__wavefunctions])

        return objective_values

    def finalize(self):
        self.__wavefunctions = self.get_wavefunctions()

        return self

    def get_wavefunctions(self) -> List[np.ndarray]:
        if self.__wavefunctions is not None:
            return self.__wavefunctions

        return [self.__get_wavefunction(step.params) for step in self.__steps]

    def get_latest_wavefunction(self):
        if self.__wavefunctions is not None:
            return self.__wavefunctions[-1]

        return self.__get_wavefunction(self.get_latest_step().params)

    def get_objective(self):
        return self.__optimiser.objective

    def get_objective_values(self):
        return [self.__optimiser.objective.evaluate(wf) for wf in self.get_wavefunctions()]

    def get_latest_objective_value(self):
        final_step = self.get_latest_step()
        wf = self.__get_wavefunction(final_step.params)

        return self.__optimiser.objective.evaluate(wf)

    def __get_wavefunction(self, params: np.ndarray) -> np.ndarray:
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
