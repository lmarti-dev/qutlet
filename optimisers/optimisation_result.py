"""
module docstring Optimisation result
"""
from typing import List
from numbers import Real

import numpy as np

from fauvqe.optimisers.optimisation_step import OptimisationStep
from fauvqe.objectives.objective import Objective


class OptimisationResult:
    """
    class docstring Optimisation result
    """

    def __init__(self, objective: Objective):
        # Store start params? optimiser.circuit_param_values
        self.__steps: List[OptimisationStep] = []
        self.__objective: Objective = objective
        self.__index = 0

    @property
    def objective(self) -> Objective:
        return self.__objective

    def add_step(self, params: np.ndarray) -> None:
        """
        Add a step
        """
        self.__steps.append(OptimisationStep(self, params=params, index=self.__index))
        self.__index += 1

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

    def get_latest_objective_value(self):
        """
        Get final value of the objective.
        """

        return self.get_latest_step().objective

    def __repr__(self) -> str:
        return "<OptimisationResult steps={} last_step={} result={} objective={}>".format(
            self.__index,
            self.get_latest_step(),
            self.get_latest_objective_value(),
            self.__objective,
        )
