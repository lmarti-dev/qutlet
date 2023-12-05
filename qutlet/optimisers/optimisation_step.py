from numbers import Real, Integral
from typing import List, Optional, Any, Literal

import numpy as np


class OptimisationStep:
    """A step of an optimisation.

    Parameters
    ----------
    parent: OptimisationResult
    index: Integral
    params: numpy.ndarray
    wavefunction: numpy.ndarray, optional
    objective: numpy.ndarray, optional
    """

    def __init__(
        self,
        parent,
        index: Integral,
        params: np.ndarray,
        wavefunction: Optional[np.ndarray] = None,
        objective: Optional[Real] = None,
    ):
        self._parent = parent
        self.index = index
        self.params = params
        self.__wavefunction: Optional[np.ndarray] = wavefunction
        self.__objective: Optional[Real] = objective

    @property
    def wavefunction(self) -> np.ndarray:
        """Get the wavefunction after the circuit of this step.

        Calculates and stores the wavefunction if it is not stored.

        Returns
        -------
        numpy.ndarray
            The wavefunction
        """
        if self.__wavefunction is None:
            self.__wavefunction = self._parent.objective.simulate(
                param_resolver=self._parent.objective.model.get_param_resolver(self.params),
            )

        return self.__wavefunction

    @property
    def objective(self) -> Real:
        """Get the objective value.

        Calculates and stores the objective if it is not stored. For that it also calculates and stores the wavefunction
        if not available.

        Returns
        -------
        Real
            The objective value
        """
        if self.__objective is None:
            self.__objective = self._parent.objective.evaluate(self.wavefunction)

        return self.__objective
    
    def reset_objective(self) -> None:
        self.__objective = None
    
    def __repr__(self) -> str:
        return "<OptimisationStep index={} params={}>".format(self.index, self.params)

    def to_list(
        self, columns: List[Literal["index", "params", "wavefunction", "objective"]]
    ) -> List:
        """Convert to step to a ordered list with the given columns.

        Parameters
        ----------
        columns: list of {"index", "params", "wavefunction", "objective"}

        Returns
        -------
        list of any
            Ordered list of the columns requested.
        """
        return [self._resolve_column(column) for column in columns]

    def _resolve_column(
        self, column: Literal["index", "params", "wavefunction", "objective"]
    ) -> Any:
        """Resolve a single column for the `.to_dict()` method.

        Parameters
        ----------
        column: {"index", "params", "wavefunction", "objective"}

        Returns
        -------
        any

        Raises
        -------
        NotImplementedError: If `column` is not supported.
        """
        if column == "index":
            return self.index

        if column == "params":
            return self.params

        if column == "wavefunction":
            return self.wavefunction

        if column == "objective":
            return self.objective

        raise NotImplementedError("Unknown column {}".format(column))  # pragma: no cover
