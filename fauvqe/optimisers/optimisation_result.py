"""
module docstring Optimisation result
"""
import datetime
import json
from numbers import Real, Integral
from typing import List, Optional

import numpy as np
from fauvqe.restorable import Restorable
from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_step import OptimisationStep


class OptimisationResult:
    """OptimisationResult class docstring"""

    def __init__(self, objective: Objective):
        """

        Parameters
        ----------
        objective
        """
        # Store start params? optimiser.circuit_param_values
        self.__steps: List[OptimisationStep] = []
        self.__objective: Objective = objective
        self.__index = 0

    def store(self, path: str, indent: Optional[Integral] = None) -> None:
        dct = {
            "result": {
                "objective": self.objective.to_json_dict(),
                "steps": [step.to_dict() for step in self.__steps],
            },
            "meta": {
                "time": datetime.datetime.now(),
            },
        }

        with open(path, "w") as outfile:
            json.dump(dct, outfile, indent=indent)
            outfile.close()

    @classmethod
    def restore(cls, path):
        with open(path, "r") as infile:
            dct = json.load(infile)
            infile.close()

            obj = Restorable.restore(dct["result"]["objective"])
            res = OptimisationResult(obj)

            for step in sorted(dct["result"]["steps"], key=lambda s: s["index"]):
                del step["index"]
                res.add_step(**step)

            return res

    @property
    def objective(self) -> Objective:
        """

        Returns
        -------

        """
        return self.__objective

    def add_step(
        self,
        params: np.ndarray,
        wavefunction: Optional[np.ndarray] = None,
        objective: Optional[Real] = None,
    ) -> None:
        """

        Parameters
        ----------
        params: numpy.ndarray
        wavefunction: numpy.ndarray optional
        objective: Real optional
        """
        self.__steps.append(
            OptimisationStep(
                self,
                params=params,
                index=self.__index,
                wavefunction=wavefunction,
                objective=objective,
            )
        )
        self.__index += 1

    def get_steps(self) -> List[OptimisationStep]:
        """

        Returns
        -------
        List[OptimisationStep]
        """
        return self.__steps

    def get_latest_step(self) -> OptimisationStep:
        """

        Returns
        -------
        OptimisationSteps
        """
        return self.__steps[-1]

    def get_latest_objective_value(self) -> Real:
        """

        Returns
        -------

        """
        return self.get_latest_step().objective

    def __repr__(self) -> str:
        return "<OptimisationResult steps={} latest_step={} objective={}>".format(
            self.__index,
            self.get_latest_step(),
            self.__objective,
        )
