"""
cVaR objective module docstring
"""

from typing import Literal, Tuple
import numpy as np
from numbers import Real

from .objective import Objective
from ..isings.initialisers import Initialiser


class CVaR(Objective):
    """Class implementing the conditional value at risk (cVaR) as an objective.

    Explantion of parameters and cVaR.

    Parameters
    ----------
    alpha: float
        The :math:`\\alpha` value
    field: {"Z"} default "Z"
        The field to be evaluated


    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <cVaR field=self.field alpha=self.alpha>
    """

    def __init__(self, alpha: float, field: Literal["Z", "X"] = "Z"):
        super().__init__()

        assert 0 <= alpha <= 1.0, "cVaR alpha must be in (0, 1). Recieved: {:f}".format(alpha)
        assert field in [
            "Z",
            "X",
        ], "Bad input 'field'. Allowed values are ['Z' (default), 'X'], revieced {}".format(field)

        self.__alpha = alpha
        self.__mask_j = None
        self.__mask_h = None
        self.__energies_j: np.ndarray = None
        self.__energies_h: np.ndarray = None

        self.__field: Literal["Z", "X"] = field
        self.__n_qubits: int = 0

    def initialise(self, obj_value: Tuple[np.ndarray]) -> None:
        """Link this cVaR instance to a given initialiser and compute the sorting mask and energies.

        - Calculate energies  (do not respect field yet)
        - sort energies (for easy reuse)
        - store both in __energies and __mask

        Parameters
        ----------
        obj_value:
            The objective values.
        """
        energies_j, energies_h = obj_value

        self.__mask_j = np.argsort(energies_j)
        self.__mask_h = np.argsort(energies_h)

        self.__energies_j = energies_j[self.__mask_j]
        self.__energies_h = energies_h[self.__mask_h]

        self.__n_qubits = np.log2(energies_h.size())

    def evaluate(self, wavefunction: np.ndarray) -> Real:
        """Calculate the conditional value at risk for a given wavefunction.

        Explain method (cumsum etc.)
        Return minimum if below

        Parameters
        ----------
        wavefunction:
            The wavefunction to calculate the conditional value at risk.

        Returns
        ----------
        numpy.float64:
            The conditional value at risk (cVaR)
        """

        # Wavefunction is expected to be normalized
        # Reorder wavefunction with stored mask
        probabilities_j = np.abs(wavefunction) ** 2
        probabilities_j = probabilities_j[self.__mask_j]

        probabilities_h = probabilities_j

        if self.__field == "X":
            wavefunction_x = self._rotate_x(wavefunction)
            probabilities_h = np.abs(wavefunction_x) ** 2
            probabilities_h = probabilities_h[self.__mask_h]

        # Calculate cVaR independently for interaction and field
        cvar_j = CVaR._calc_cvar(probabilities_j, self.__energies_j, self.__alpha)
        cvar_h = CVaR._calc_cvar(probabilities_h, self.__energies_h, self.__alpha)

        if self.__field == "X":
            return (-cvar_h - cvar_j) / self.__n_qubits

        if self.__field == "Z":
            return (-cvar_j + cvar_h) / self.__n_qubits

    @staticmethod
    def _calc_cvar(probabilities: np.ndarray, energies: np.ndarray, alpha: float) -> Real:
        # Expect energies and probabilities to be ordered
        mask = np.cumsum(probabilities) <= alpha

        if not np.any(mask):
            mask = energies == np.min(energies)
            # Todo: This may not be unique! Use all values at minimum
            # return energies[0]

        energies_ = energies[mask]
        probabilities_ = probabilities[mask]

        return np.sum(energies_ * probabilities_) / np.sum(probabilities_)

    def __repr__(self) -> str:
        return "<cVaR field={} alpha={}>".format(self.__field, self.__alpha)
