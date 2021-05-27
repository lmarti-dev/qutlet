"""
module docstring
"""
from numbers import Real, Integral
from typing import Union, List, Optional, Any

import numpy as np


class OptimisationStep:
    """

    Parameters
    ----------
    parent: OptimisationResult
    index: Integral
    params: numpy.ndarray
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
        """

        Returns
        -------

        """
        if self.__wavefunction is None:
            self.__wavefunction = self._parent.objective.simulate(
                param_resolver=self._parent.objective.model.get_param_resolver(self.params),
            )

        return self.__wavefunction

    @property
    def objective(self) -> Real:
        """

        Returns
        -------

        """
        if self.__objective is None:
            self.__objective = self._parent.objective.evaluate(self.wavefunction)

        return self.__objective

    def __repr__(self) -> str:
        return "<OptimisationStep index={} params={}>".format(self.index, self.params)

    def to_list(self, columns: List[str]) -> List:
        return [self._resolve_column(column) for column in columns]

    def _resolve_column(self, column: str) -> Any:
        if column == "index":
            return self.index

        if column == "params":
            return self.params.tolist()

        if column == "wavefunction":
            return self.wavefunction.tolist()

        if column == "objective":
            return self.objective

        raise NotImplementedError("Unknown column {}".format(column))
