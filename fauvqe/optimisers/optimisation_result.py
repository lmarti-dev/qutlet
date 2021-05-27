"""
module docstring Optimisation result
"""
import datetime
from numbers import Real, Integral
from typing import List, Optional, Literal, Union
import pathlib

import numpy as np
import fauvqe.json as json
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

    def store(
        self,
        path: Union[pathlib.Path, str],  # Todo: Support File-like objects
        #  (see: https://docs.python.org/3/library/io.html#io.TextIOBase)
        indent: Optional[Integral] = None,
        store_wavefunctions: Literal["none", "available", "all"] = "none",
        store_objectives: Literal["none", "available", "all"] = "none",
    ) -> None:
        allowed_store = ["none", "available", "all"]
        assert (
            store_objectives in allowed_store
        ), "store_objectives must be one of {}, default: 'none'. {} was given".format(
            allowed_store, store_objectives
        )
        assert (
            store_wavefunctions in allowed_store
        ), "store_wavefunctions must be one of {}, default: 'none'. {} was given".format(
            allowed_store, store_wavefunctions
        )
        assert not (store_objectives == "all" and store_wavefunctions == "available"), (
            "Unwanted side effects: with store_objectives='all'"
            + ", store_wavefunctions='available' is equivalent to store_wavefunctions='all'"
        )

        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        assert not path.exists(), "Not overwriting existing path {}".format(path)

        # Calculate all wavefunction/objectives if not available
        if store_wavefunctions == "all":
            self.get_wavefunctions()
        if store_objectives == "all":
            self.get_objectives()

        # Collect columns to store
        columns = ["index", "params"]
        if store_wavefunctions != "none":
            columns.append("wavefunction")
        if store_objectives != "none":
            columns.append("objective")

        dct = {
            "result": {
                "objective": self.objective.to_json_dict(),
                "steps": {
                    "cols": columns,
                    "rows": [step.to_list(columns) for step in self.__steps],
                },
            },
            "meta": {
                "time": str(datetime.datetime.now()),
            },
        }

        with path.open("w") as outfile:
            json.dump(dct, outfile, indent=indent)
            outfile.close()

    @classmethod
    def restore(cls, path: Union[pathlib.Path, str]):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        assert path.exists(), "Cannot load nonexistent path {}".format(path)

        with path.open("r") as infile:
            dct = json.load(infile)
            infile.close()

            obj = Restorable.restore(dct["result"]["objective"])
            res = OptimisationResult(obj)

            # Convert steps to dict (keep this verbose)
            steps = []
            step_cols = dct["result"]["steps"]["cols"]
            for step_row in dct["result"]["steps"]["rows"]:
                steps.append({step_cols[i]: step_row[i] for i in range(len(step_cols))})

            # Add steps sorted by their index
            for step in sorted(steps, key=lambda s: s["index"]):
                del step["index"]  # Remove index from dict
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

    def get_wavefunctions(self) -> List[np.ndarray]:
        # For docs: This has huge side effects (floods RAM)
        # Todo: optimise with joblib?
        return [step.wavefunction for step in self.__steps]

    def get_objectives(self) -> List[Real]:
        # For docs: This has huge side effects (floods RAM)
        # Note: this method is incompatible with making optimisation parameters dependant on step
        return [self.objective.evaluate(wavefunction) for wavefunction in self.get_wavefunctions()]

    def __repr__(self) -> str:
        return "<OptimisationResult steps={} latest_step={} objective={}>".format(
            self.__index,
            self.get_latest_step(),
            self.__objective,
        )
