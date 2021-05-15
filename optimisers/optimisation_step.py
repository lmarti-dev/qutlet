"""
module docstring
"""
from numbers import Real
from typing import Union

import numpy as np


class OptimisationStep:
    """
    class docstring
    """

    def __init__(self, parent, index: int, params: np.ndarray):
        self._parent = parent
        self.index = index
        self.params = params
        self.__wavefunction: Union[np.ndarray, None] = None
        self.__objective: Union[Real, None] = None

    @property
    def wavefunction(self) -> np.ndarray:
        if self.__wavefunction is None:
            self.__wavefunction = self._parent.objective.simulate(
                param_resolver=self._parent.objective.initialiser.get_param_resolver(self.params),
            )

        return self.__wavefunction

    @property
    def objective(self) -> Real:
        if self.__objective is None:
            self.__objective = self._parent.objective.evaluate(self.wavefunction)

        return self.__objective

    def __repr__(self) -> str:
        return "<OptimisationStep index={} params={}>".format(self.index, self.params)
