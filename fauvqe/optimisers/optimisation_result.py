import datetime
from numbers import Real, Integral
from typing import List, Optional, Literal, Union
import pathlib

import numpy as np
import timeit
import cirq
import joblib
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool

import fauvqe.json
from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_step import OptimisationStep


class OptimisationResult:
    def __init__(self, objective: Objective):
        # Store start params? optimiser.circuit_param_values
        self.__steps: List[OptimisationStep] = []
        self.__objective: Objective = objective
        self.__index = 0

    def store(
        self,
        path: Union[pathlib.Path, str],  # Todo: Support File-like objects
        #  (see: https://docs.python.org/3/library/io.html#io.TextIOBase)
        indent: Optional[Integral] = None,
        overwrite: bool = False,
        store_wavefunctions: Literal["none", "available", "all"] = "none",
        store_objectives: Literal["none", "available", "all"] = "none",
    ) -> None:
        """Store this `OptimisationResult` as a json file.

        Parameters
        ----------
        path: pathlib.Path or str
            The path to store the file to
        indent: Integral, optional
            Indent the json file for better human readability (default: No indent)
        overwrite: bool, default False
            Overwrite existing files
        store_wavefunctions: {"none", "available", "all"}, default "none"
            Include wavefunctions in the file.

            - "none": no wavefunction will be stored
            - "available": already calculated wavefunctions will be stored
            - "all": all wavefunctions will be calculated and then stored
        store_objectives: {"none", "available", "all"}, default "none"
            Include wavefunctions in the file.

            - "none": no objective will be stored
            - "available": already calculated objectives will be stored
            - "all": all objectives will be calculated and then stored

        Raises
        ---------
        FileExistsError
            If the desired path exists and `overwrite` is not set to True.
        """
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

        # Never overwrite existing paths
        if not overwrite and path.exists():
            raise FileExistsError("Not overwriting existing path {}".format(path))

        t0 = timeit.default_timer()
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

        t1 = timeit.default_timer()
        dct = {
            "result": {
                "objective": self.objective,
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
            fauvqe.json.dump(dct, outfile, indent=indent)
            outfile.close()
        t2 = timeit.default_timer()
        print(
            "Store, get objective value: {}s,\t write to json {}s".format(
                t1 - t0, t2 - t1
            )
        )

    def storetxt(
        self,
        path: Union[pathlib.Path, str],  # Todo: Support File-like objects
        overwrite: bool = False,
        additional_objectives: List[Objective] = None,
    ) -> None:
        """
        Store Objective values of ths `OptimisationResult` in a txt file.

        Parameters
        ----------
        path: pathlib.Path or str
            The path to store the file to
        overwrite: bool, default False
            Overwrite existing files
        additional_objectives: List[Objective]
            Calculate and include objective values of the given additional objectives in txt file


        Raises
        ---------
        FileExistsError
            If the desired path exists and `overwrite` is not set to True.
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        # Never overwrite existing paths
        if not overwrite and path.exists():
            raise FileExistsError("Not overwriting existing path {}".format(path))

        header_string = "{} \t".format(self.__objective.__class__.__name__)
        if additional_objectives is None:
            save_data = self.get_objectives()
        else:
            if isinstance(additional_objectives, Objective):
                additional_objectives = [additional_objectives]
            save_data = np.empty([self.__index, 1 + len(additional_objectives)])
            save_data[:, 0] = self.get_objectives()

            i = 1
            for objective in additional_objectives:
                header_string += "{} \t".format(objective.__class__.__name__)
                save_data[:, i] = self.get_objectives(objective)
                i += 1

        np.savetxt(path, save_data, header=header_string, delimiter="\t")

    @classmethod
    def restore(cls, path: Union[pathlib.Path, str]) -> "OptimisationResult":
        """Restore a previously stored `OptimisationResult`.

        Parameters
        ----------
        path: pathlib.Path or str
            The path where to find the stored `OptimisationResult`.

        Returns
        -------
        OptimisationResult
            The restored result.

        Raises
        -------
        FileNotFoundError
            If the path is not a file
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError("Cannot load nonexistent path {}".format(path))

        with path.open("r") as infile:
            dct = fauvqe.json.load(infile)
            infile.close()

            obj = dct["result"]["objective"]
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
        """The optimised `Objective`

        Returns
        -------
        Objective
            The optimised `Objective`
        """
        return self.__objective

    def add_step(
        self,
        params: np.ndarray,
        wavefunction: Optional[np.ndarray] = None,
        objective: Optional[Real] = None,
    ) -> None:
        """Add a step to the optimisation result.

        Parameters
        ----------
        params: numpy.ndarray
        wavefunction: numpy.ndarray, optional
        objective: Real, optional
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
        """Get all steps added to the result.

        Returns
        -------
        list of OptimisationStep
        """
        return self.__steps

    def get_latest_step(self) -> OptimisationStep:
        """Get the latest step.

        Returns
        -------
        OptimisationSteps
        """
        return self.__steps[-1]

    def get_latest_objective_value(self) -> Real:
        """Get the latest objective value.

        Returns
        -------
        Real
        """
        return self.get_latest_step().objective

    def get_wavefunctions(self) -> List[np.ndarray]:
        """Get all wavefunctions.

        Returns
        -------
        list of numpy.ndarray
        """
        # Todo: optimise with joblib?
        return [step.wavefunction for step in self.__steps]

    def _get_wf_from_i(self, i) -> np.ndarray:
        return self.__steps[i].wavefunction

    def get_objectives(self, objective: Optional[Objective] = None) -> List[Real]:
        """Get all objective values.

        Notes
        -------
        Watch out that this may flood RAM as also it needs to calculate the wavefunctions and does not explicitly delete
        them.

        Returns
        -------
        list of Real
        """
        if objective == None:
            objective = self.objective

        _n_jobs = 8

        pool = Pool(_n_jobs - 1)
        _wfs = pool.map(self._get_wf_from_i, range(self.__index))

        _objective_values = joblib.Parallel(n_jobs=_n_jobs, backend="threading")(
            joblib.delayed(objective.evaluate)(_wfs[i]) for i in range(len(_wfs))
        )

        return _objective_values

    def __repr__(self) -> str:
        return "<OptimisationResult steps={} latest_step={} objective={}>".format(
            self.__index,
            self.get_latest_step(),
            self.__objective,
        )
